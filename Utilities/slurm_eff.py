#!/usr/bin/env python3
"""
Slurm Efficiency Reporting Tool

This program analyzes Slurm job accounting data from the MySQL/MariaDB database
to calculate resource efficiency metrics (CPU, memory, GPU) for HPC users.
Monthly efficiency reports are generated and emailed to users.

Author: Larry N. Singh (larryns/singhln)
Date: 2026-01-10
"""

import boto3
import logging
import json
import argparse
import csv
import gzip
import smtplib

from sqlalchemy import create_engine, select, func, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from datetime import datetime
from enum import Enum
from string import Template
from typing import Optional

from email.message import EmailMessage


# SLURM enumerated types
class JobStates(Enum):
    JOB_PENDING = 0         # queued waiting for initiation
    JOB_RUNNING = 1         # allocated resources and executing
    JOB_SUSPENDED = 2       # allocated resources, execution suspended
    JOB_COMPLETE = 3        # completed execution successfully
    JOB_CANCELLED = 4       # cancelled by user
    JOB_FAILED = 5          # completed execution unsuccessfully
    JOB_TIMEOUT = 6         # terminated on reaching time limit
    JOB_NODE_FAIL = 7       # terminated on node failure
    JOB_PREEMPTED = 8       # terminated due to preemption
    JOB_BOOT_FAIL = 9       # terminated due to node boot failure
    JOB_DEADLINE = 10       # terminated on deadline
    JOB_OOM = 12            # experienced out of memory error


# From slurm/src/common/slurmdb_defs.h
class TresTypes(Enum):
    TRES_CPU = 1
    TRES_MEM = 2
    TRES_ENERGY = 3
    TRES_NODE = 4
    TRES_BILLING = 5
    TRES_FS_DISK = 6
    TRES_VMEM = 7
    TRES_PAGES = 8
    TRES_STATIC_CNT = 9


# This list corresponds to the tres_table in slurmdb. We could pull it from slurmdb
# but I decided to hardcode it here because I'm lazy, and it's unlikely to change.
class TresTable(Enum):
    GRES_GPU = 1001
    GRES_GPUMEM = 1002
    GRES_GPUUTIL = 1003


# Flags used for computing memory usage.
# Slurm sets bit 63 to 1 to indicate memory per cpu
MEM_PER_CPU_FLAG = (1 << 63)

# Mask for the remaining 63 bits
MEM_MASK = (MEM_PER_CPU_FLAG - 1)

# Mail settings constants. We are going to use lazy evaluation formatting.
MSG_FROM = 'HPC Team <HPCTeam@domain.edu>'
MSG_SUBJECT = '{cluster} Slurm Monthly Resource Efficiency Report'
MSG_TO = '{userid}@domain.edu'
RELAY_HOST = 'your.relay.mailer.host.here'


# Class for sending e-mail to users
class SMTPMailer:
    def __init__(
            self, 
            smtp_server: str, 
            smtp_port: int = 25,
            timeout: int = 10,
            debug: bool = False,
            use_starttls: bool = True
            ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.timeout = timeout
        self.debug = debug
        self.use_starttls = use_starttls

    def send_email(
            self, 
            from_addr: str, 
            to_addr: str, 
            subject: str, 
            body: str,
            attachment: Optional[str] = None):
        # Build the message
        msg = EmailMessage()
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg['Subject'] = subject
        msg.set_content(body)

        # Check if we have an attachment
        if attachment is not None:
            try:
                with open(attachment, 'rb') as f:
                    file_data = f.read()
                    file_name = attachment.split('/')[-1]

                    # Check for gzip files
                    if file_name.endswith('.gz'):
                        subtype = 'gzip'
                    else:
                        subtype = 'octet-stream'

                    msg.add_attachment(
                        file_data,
                        maintype='application',
                        subtype=subtype,
                        filename=file_name
                    )
            except FileNotFoundError:
                logger.error(f"Attachment file {attachment} not found.")
            except PermissionError:
                logger.error(f"Permission denied for attachment file {attachment}.")
            except Exception as e:
                logger.error(f"Failed to attach file {attachment}: {e}")

        # Message built, now send
        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
            server.set_debuglevel(1 if self.debug else 0)

            server.ehlo()
            if self.use_starttls:
                server.starttls()
                server.ehlo()

            logger.info(f"Sending e-mail to {to_addr} via {self.smtp_server}:{self.smtp_port}")
            server.send_message(msg)
            logger.info(f"E-mail sent successfully to {to_addr}.")
            

EmailTemplate = Template("""
Hello $user,

Your Skyline resource efficiences have been calculated for the last month. These
results are calculated as a time weighted average of what you reserved for your jobs
and what you actually used.

CPU Efficiency: $cpu_eff%
Memory Efficiency: $mem_eff%

You ran ${num_jobs} jobs for total wall time of ${total_walltime} seconds.

Adjustments to your jobs will help improve the overall efficiency of the Skyline cluster
for all users. You can get individual job efficiency at any times using `seff -j <jobid>`.
You can modify a queued job as folows:
> scontrol update jobid=<jobid> minmemorymode=<memory in megabytes>
> scontrol update jobid=<jobid> mincpusnode=<cpus>

Attached is a list of your jobs and their individual efficiencies for the month 
(limitited to the first 10,000 jobs).

Note that you may see small differences in efficiency between these files and the 
seff command due to how round off is handled.  Please contact the HPC team by e-mailing
HPCTeam@domain.edu if you have any questions.

Thank you for using the Skyline cluster.
HPC Team
""")

# Configure logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# We are going to pull the password from the secret manager. This code is pulled from AWS:
# https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python-sdk.

# You'll have to run spaces creds --account oeb-hpc-dev -role Admin
# to get new creds.


class GetSecretWrapper:
    def __init__(self, secretsmanager_client):
        self.client = secretsmanager_client

    def get_secret(self, secret_name):
        """
        Retrieve individual secrets from AWS Secrets Manager using the get_secret_value API.
        This function assumes the stack mentioned in the source code README has been successfully deployed.
        This stack includes 7 secrets, all of which have names beginning with "mySecret".

        :param secret_name: The name of the secret fetched.
        :type secret_name: str
        """
        try:
            get_secret_value_response = self.client.get_secret_value(
                SecretId=secret_name
            )
            logging.info("Secret retrieved successfully.")
            return get_secret_value_response["SecretString"]
        except self.client.exceptions.ResourceNotFoundException:
            msg = f"The requested secret {secret_name} was not found."
            logger.info(msg)
            return msg
        except Exception as e:
            logger.error(f"An unknown error occurred: {str(e)}.")
            raise


# Pull the password from AWS Secrets
secret_name = "rds!db-2a71c645-d712-407f-b735-e04fa3ede699"
secret = None

try:
    # Validate secret name
    if not secret_name:
        raise ValueError("Even the secret name is a secret. Enter a secret name, please!")

    # Get the secret by name
    aws_session = boto3.Session(profile_name="oeb-hpc-dev-admin", region_name='us-east-1')
    client = aws_session.client("secretsmanager")
    wrapper = GetSecretWrapper(client)
    secret_data = wrapper.get_secret(secret_name)
except Exception as e:
    logging.error(f"Error retrieving secret: {e}")
    raise

# The secret is stored as a json entry
secret = json.loads(secret_data)

# Create the engine/connection to the slurm db
conf = {
    'host': "slurm-scheduler.cv7fqg4tqh6d.us-east-1.rds.amazonaws.com",
    'port': "3306",
    'database': "database1",
    'user': secret["username"],
    'password': secret["password"]
}


# Now connect to the database and create a session
engine = create_engine(
    "mysql+mysqldb://{user}:{password}@{host}:{port}/{database}".format(**conf)
)

Session = sessionmaker(bind=engine)
session = Session()

# We're using the ORM, and going to automatically gneerate mapped classes and relationships
Base = automap_base()
Base.prepare(autoload_with=engine)


def user_month_eff(session, year: int, month: int, cluster: str):
    """Compute the CPU efficiency for a user over a given month.
    """

    # Reflect to get the table schemas we need
    JobTable = Base.classes[f"{cluster}_job_table"]
    AssocTable = Base.classes[f"{cluster}_assoc_table"]
    StepTable = Base.classes[f"{cluster}_step_table"]

    # Determine boundaries of the month
    start_dt = datetime(year, month, 1)
    end_dt = datetime(year + (month == 12), (month % 12) + 1, 1)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    # The following code tries to emulate the code from seff:
    # https://github.com/SchedMD/slurm/tree/4b51ce3796d59fbd1dd5c3fe151c7fdc1b999934/contribs/seff
    # Look also in the perl/Slurmdb.xs for the C interface to perl for accessing the slurmdb.
    job_elapsed = func.cast(
        func.greatest(JobTable.time_end - JobTable.time_start - JobTable.time_suspended, 0),
        Integer
    )

    # total core walltime
    corewalltime = job_elapsed * JobTable.cpus_req

    # compute cpu time
    tot_cpu_sec = func.sum(StepTable.user_sec) + func.sum(StepTable.sys_sec)
    tot_cpu_usec = func.sum(StepTable.user_usec) + func.sum(StepTable.sys_usec)
    cput = func.cast(tot_cpu_sec + func.round(tot_cpu_usec / 1000000, 0), Integer)

    # cpu efficiency
    cpu_eff = func.round(cput / corewalltime * 100.0, 2)

    # functions for computing the memory efficiency.

    # We need to compute the max memory used over all steps of the job. Some
    # steps may be NULL, so we will use coalesce to convert NULL to 0. The memory units
    # are kb. The TRES string is a comma separated listof key=value pairs. We will extract
    # the memory usage using the following regex.
    mem_pattern = f".*(^|,){TresTypes.TRES_MEM.value}=([0-9]+).*"
    step_mem = func.regexp_replace(StepTable.tres_usage_in_tot, mem_pattern, '\\2')
    mem = func.round(func.max(func.cast(step_mem, Integer)) / 1024 / 1024 / 1024, 2)  # in GB

    # Now let's find the allocated memory.
    mem_per_cpu_flag = JobTable.mem_req.bitwise_and(MEM_PER_CPU_FLAG) != 0
    mem_mb = JobTable.mem_req.bitwise_and(MEM_MASK)

    # if set memory is per cpu otherwise it's per node
    if mem_per_cpu_flag:
        mem_mb = mem_mb * JobTable.cpus_req

    alloc_mem = func.round(mem_mb / 1024, 2)  # In GB

    # Compute memory efficiency
    memeff = func.round(mem * 100.0 / alloc_mem, 2)

    # Okay now let's compute the GPU usage and effiency.

    # First define a pattern for the number of allocated gpus. MariaDB uses PCRE so we will
    # use a positive lookbehind to find the value after the GRES_GPU key.
    num_gpus_pattern = f"(?<={TresTable.GRES_GPU.value}=)[0-9]+"
    num_gpus = func.cast(func.regexp_substr(JobTable.tres_alloc, num_gpus_pattern), Integer)

    # Now get the gpu utilization
    gpu_util_pattern = f"(?<={TresTable.GRES_GPUUTIL.value}=)[0-9]+"
    gpu_util = func.sum(
        func.cast(func.regexp_substr(StepTable.tres_usage_in_tot, gpu_util_pattern), Integer)
    )

    # And the gpu memory usage
    gpu_mem_pattern = f"(?<={TresTable.GRES_GPUMEM.value}=)[0-9]+"
    gpu_mem = func.round(
        func.max(
            func.cast(func.regexp_substr(StepTable.tres_usage_in_tot, gpu_mem_pattern), Integer)
        ) / 1024 / 1024 / 1024,
        2
    )

    # Build the query
    stmt = (
        select(
            AssocTable.user.label("username"),        # login name from assoc table
            JobTable.id_job.label("id_job"),
            job_elapsed.label("job_elapsed"),
            cput.label("cput"),
            corewalltime.label("corewalltime"),
            mem.label("mem"),
            cpu_eff.label("cpu_eff"),
            alloc_mem.label("alloc_mem"),
            memeff.label("memeff"),
            num_gpus.label("num_gpus"),
            gpu_util.label("gpu_util"),
            gpu_mem.label("gpu_mem")
        )
        # Job → Assoc (for username + filter)
        .join(AssocTable, JobTable.id_assoc == AssocTable.id_assoc)
        # Job → Step (to accumulate CPU usage)
        .join(StepTable, StepTable.job_db_inx == JobTable.job_db_inx)
        .where(
            JobTable.state == JobStates.JOB_COMPLETE.value,
            JobTable.time_end >= start_ts,
            JobTable.time_end < end_ts,
            job_elapsed > 0,
            AssocTable.user.in_(['singhln'])  # For testing only; remove or modify for production
        )
        .group_by(
            AssocTable.user,
            JobTable.job_db_inx
        )
        .order_by(AssocTable.user)
    )

    return session.execute(stmt).all()


def email(current_user, total_cput, total_corewalltime, total_mem, total_alloc_mem, num_jobs, mailer, filename, cluster):
    # Before we update the new user info, send the e-mail
    # Compute the totals and build the e-mail body
    total_cpu_eff = round(total_cput * 100 / total_corewalltime, 2) if total_corewalltime > 0 else 0.0
    total_mem_eff = round(total_mem * 100 / total_alloc_mem, 2) if total_alloc_mem > 0 else 0.0

    # Create the e-mail body
    mailer_body = EmailTemplate.substitute(
        user=current_user,
        cpu_eff=total_cpu_eff,
        mem_eff=total_mem_eff,
        total_walltime=total_corewalltime,
        num_jobs=num_jobs
    )

    # send the e-mail
    mailer.send_email(
        from_addr=MSG_FROM,
        to_addr=MSG_TO.format(userid=current_user),
        subject=MSG_SUBJECT.format(cluster=cluster),
        body=mailer_body,
        attachment=filename
    )

def send_eff(results, year, month, output_dir, max_rows_per_user, cluster):
    """
    Write query results to separate CSV files, one per user.
    Results must be ordered by username.

    :param results: List of named tuples from user_month_eff() (ordered by username)
    :param year: Year for filename
    :param month: Month for filename
    :param output_dir: Directory to write CSV files
    :param cluster: Cluster name for email subject
    """
    fieldnames = [
        'JobId', 'ElapsedTime', 'TotalCPUTime', 'TotalWalltime',
        'CPUEfficiency', 'MemoryUsedGB', 'AllocatedMemoryGB', 'MemEfficiency',
        'NumGPUs', 'TotalGPUTime', 'GPUMemoryGB'
    ]

    current_user = None
    csvfile = None
    writer = None
    row_count = 0
    write_row_count = 0

    # Used to compute the overall cpu efficiency
    total_cput = 0
    total_corewalltime = 0

    # Used to compute the overall memory efficiency
    total_mem = 0
    total_alloc_mem = 0

    filename = None

    # Reset counters for new users
    def reset_user_counters():
        nonlocal total_cput, total_corewalltime
        nonlocal total_mem, total_alloc_mem
        nonlocal row_count, write_row_count

        total_cput = 0
        total_corewalltime = 0
        total_mem = 0
        total_alloc_mem = 0
    
        row_count = 0
        write_row_count = 0


    # Build the email object
    mailer = SMTPMailer(RELAY_HOST)

    for row in results:
        # When username changes, close previous file and open new one
        if row.username != current_user:
            if csvfile:
                csvfile.close()
                logger.info(f"Wrote {write_row_count} of {row_count} rows for user '{current_user}' to {filename}")

                # send the e-mail
                email(
                    current_user, total_cput, total_corewalltime, total_mem,
                    total_alloc_mem, row_count, mailer, filename, cluster
                )

            # Now we can update the current user
            current_user = row.username
            filename = f"{output_dir}/{current_user}_{year}_{month:02d}.csv.gz"
            csvfile = gzip.open(filename, 'wt', newline='')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Reset counters
            reset_user_counters()

        # Write the row
        if write_row_count < max_rows_per_user:
            writer.writerow({
                'JobId': row.id_job,
                'ElapsedTime': row.job_elapsed,
                'TotalCPUTime': row.cput,
                'TotalWalltime': row.corewalltime,
                'CPUEfficiency': row.cpu_eff,
                'MemoryUsedGB': row.mem,
                'AllocatedMemoryGB': row.alloc_mem,
                'MemEfficiency': row.memeff,
                'NumGPUs': row.num_gpus,
                'TotalGPUTime': row.gpu_util,
                'GPUMemoryGB': row.gpu_mem
            })
            write_row_count += 1

        # Update the total counts
        total_cput += row.cput
        total_corewalltime += row.corewalltime
        total_mem += row.mem
        total_alloc_mem += row.alloc_mem

        row_count += 1

    # Close the last file
    if csvfile:
        csvfile.close()
        logger.info(f"Wrote {write_row_count} of {row_count} rows for user '{current_user}' to {filename}")

        # Send the last e-mail here
        email(
            current_user, total_cput, total_corewalltime, total_mem,
            total_alloc_mem, row_count, mailer, filename, cluster
        )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute SLURM efficiency for a specific month and year.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'year',
        type=int,
        help='Year (2000-2050)'
    )

    parser.add_argument(
        'month',
        type=int,
        help='Month (1-12)'
    )

    parser.add_argument(
        'cluster',
        type=str,
        help=('Cluster name. This will be the prefix used for the slurmdb database tables.')
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Output directory for CSV files'
    )

    parser.add_argument(
        '-n', '--csvlimit',
        type=int,
        default=10000,
        help="Maximum number of rows to write per user CSV file"
    )

    args = parser.parse_args()

    # Validate year range
    if args.year < 2000 or args.year > 2050:
        parser.error(f"Year must be between 2000 and 2050, got {args.year}")

    # Validate month range
    if args.month < 1 or args.month > 12:
        parser.error(f"Month must be between 1 and 12, got {args.month}")

    # Get the results
    rows = user_month_eff(session, args.year, args.month, args.cluster)

    # Write to file
    send_eff(rows, args.year, args.month, args.directory, args.csvlimit, args.cluster)

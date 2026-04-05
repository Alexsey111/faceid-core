import logging
import sys


class JobIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "job_id=%(job_id)s "
            "message=%(message)s"
        ),
        stream=sys.stdout,
        force=True,
    )
    root_logger = logging.getLogger()
    job_id_filter = JobIdFilter()
    root_logger.addFilter(job_id_filter)
    for handler in root_logger.handlers:
        handler.addFilter(job_id_filter)

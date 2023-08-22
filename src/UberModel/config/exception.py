import sys


def error_message_detail(error, error_details: sys):
    if error_details == "":
        return f"Error message {str(error)}"
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)  # inheriting init
        self.error_message = error_message_detail(
            error=error_message, error_details=error_details)

    def __str__(self) -> str:
        return self.error_message
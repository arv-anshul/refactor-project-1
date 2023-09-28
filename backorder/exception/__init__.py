from types import TracebackType
from typing import TypeAlias

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]


class BackorderException(Exception):
    def __init__(self, error_message: object, error_detail: OptExcInfo):
        super().__init__(error_message)
        self.error_message = BackorderException.get_detailed_error_message(
            error_message=error_message, error_detail=error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message: object, error_detail: OptExcInfo) -> str:
        """
        error_message: Exception object
        error_detail: object of sys module
        """
        try:
            _, _, exec_tb = error_detail
            exception_block_line_number = exec_tb.tb_frame.f_lineno  # type: ignore
            file_name = exec_tb.tb_frame.f_code.co_filename  # type: ignore
        except Exception:
            exception_block_line_number = -1
            file_name = "Unknown"

        error_message = f"[{file_name}]:[{exception_block_line_number}] - {error_message}]"
        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return BackorderException.__name__

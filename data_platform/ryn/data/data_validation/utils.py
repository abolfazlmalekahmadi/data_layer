import time
from functools import wraps

def time_validator(func):
    """
    A decorator that times the execution of a validator's `validate` method.
    It captures the result of the validation, adds a 'duration_seconds'
    attribute to it, and then returns the modified result.
    """
    @wraps(func)  # Preserves the original function's metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        # 1. Record the start time
        start_time = time.perf_counter()
        
        # 2. Call the original validate method (e.g., TabularValidator.validate)
        #    This will return a ValidationResult object.
        validation_result = func(*args, **kwargs)
        
        # 3. Record the end time and calculate the duration
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 4. Add the duration to the result object
        #    We will add this field to the ValidationResult dataclass next.
        validation_result.duration_seconds = duration
        
        # 5. Return the modified result object
        return validation_result
        
    return wrapper
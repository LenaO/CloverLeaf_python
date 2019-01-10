import json

# noinspection PyClassHasNoInit
class JSONEncoder(json.JSONEncoder):
    """
    Since the data may contain types like int64, which are not JSON serializable, we use this little encoder
    """
    def default(self, obj):
        """
        Serialize object.
        """
        from datetime import datetime
        import numpy as np
        from Timer import Timer,GPUTimer
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.timestamp()
        elif isinstance(obj, Timer):
            return {
                "start_time": obj.start_time,
                "end_time": obj.end_time,
                "elapsed_time": obj.elapsed_time
            }

        elif isinstance(obj, GPUTimer):
            return {
                "GPU_time": obj.elapsed_time
            }
 
        else:
            super(JSONEncoder, self).default(obj)




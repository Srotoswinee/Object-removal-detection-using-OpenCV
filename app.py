from configparser import ConfigParser
def load_config():
    config_object = ConfigParser()
    config_object.read("config.ini")
    parameters = config_object["settings"]
    debug_mode = parameters.getboolean("debug")
    video_path = parameters['video_path']
    rtsp_url = parameters['rtsp_url']
    config = {
        'camera_index': int(parameters['camera_index']),
        'frame_width': int(parameters['frame_width']),
        'frame_height': int(parameters['frame_height']),
        'model_path': parameters['model_path'],
        'skip_frames': int(parameters['skip_frames']),
        'video_path': video_path,
        'rtsp_url': rtsp_url,
        'debug_mode': debug_mode,
    }
    return config

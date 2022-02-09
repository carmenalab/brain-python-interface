import requests
import urllib3
import json

USERNAME = 'python'
PASSWORD = 'python'
WATCHTOWERURL = 'https://localhost:4343'
CAMERA_SID_LIST = ['e3v822d','e3v821f','e3v821b']
IFACE = ''
CONFIG = '480p15'
CODEC = 'H264'
ANNOTATION = 'None'
SEGTIME = '30m'

class E3VisionInterface(object):
    """E3VisionInterface

    White-Matter e3Vision camera system interface class.

    Currently hard-coded to the aoLab booth camera setup.

    Example use:

    e3vi = E3VisionInterface()
    e3vi.configure_cameras()
    e3vi.start_rec()
    e3vi.stop_rec()

    """
    def __init__(self,session_dict={}): #TODO: fit this into BMI3D
        self.username = USERNAME
        self.password = PASSWORD
        self.wt_url = WATCHTOWERURL
        self.camera_sid_list = CAMERA_SID_LIST
        self.create_session_subdir(session_dict)
        self.iface = IFACE
        self.config = CONFIG
        self.codec = CODEC
        self.anno = ANNOTATION
        self.segtime = SEGTIME
        urllib3.disable_warnings(
            urllib3.exceptions.InsecureRequestWarning
        )
        self.apitoken = self.get_api_token()
        self.bind_cameras()
        self.connect_cameras()

    def _create_session_subdir(self,session_dict): #TODO: Make this real
        """_create_session_subdir

        Creates a subdirectory string for the current session. All video files from this object are saved to this subdirectory.

        Args:
            session_dict (dict): BMI3D session dictionary.
        """
        self.subdir = 'asdf'

    def api_request(self,api_call_str,**kwargs): #TODO: This is completely bare. Put some exception handling on this vis a vis HTTP code (401, 404, etc)
        """api_request

        Wrapper method for all e3Vision Watchtower HTTP RESTful API requests.

        Args:
            api_call_str (str): HTTP API request string. Defines request type. Examples: '/api/login', '/api/cameras/action'
            **kwargs: keyword and value pairs populating request data dict

        Returns:
            r (request.Request): API call return in request.Request format.
        """
        api_call = self.watchtowerurl + api_call_str
        data = {}
        for k, v in kwargs.items():
            data_key = k + '[]' if isinstance(v,list) else k
            data[data_key] = v
        data['apitoken'] = self.apitoken
        return requests.post(
            api_call,
            data = data,
            verify = False,
        )

    def api_login(self):
        """get_api_token

        Opens an API session with Watchtower. Returns the API auth token for the current session.

        Returns:
            apitoken (str): auth token for the current session. 
        """
        r = self.api_request(
            '/api/login',
            username=self.username,
            password=self.password)
        j = json.loads(r.text)
        return j['apitoken']

    def configure_cameras(self):
        """configure_cameras

        Binds, syncs and connects to all cameras in interface serial ID list using the desired configuration.
        """
        for cs in self.camera_sid_list:
            self._bind_camera(cs)
            self._update_sync(cs)
            self._connect_camera(cs)

    def start_rec(self):
        """start_rec

        Begin recording a video file to the session subdirectory. Records from all connected cameras simulataneously to separate files.
        File names have the following form: <global_dir>/<session_subdir>/[cameraname]-[starttime]-[endtime].[avi | mp4]
        """
        self.api_request(
            '/api/cameras/action',
            SerialGroup=self.camera_sid_list,
            Action='RECORDGROUP',
            AdditionalPath=self.subdir,
        )

    def stop_rec(self):
        """stop_rec

        Stop all current video file recordings.
        """
        self.api_request(
            '/api/cameras/action',
            SerialGroup=self.camera_sid_list,
            Action='STOPRECORDGROUP',
        )

    def _bind_camera(self,cs):
        """_bind_camera

        Binds the camera with serial ID cs. This establishes a secure authenticated session with the camera.

        Args:
            cs (str): Camera serial ID number.
        """
        self.api_request(
            '/api/cameras/action',
            Serial=cs,
            Action='BIND',
        )

    def _update_sync(self,cs):
        """_update_sync

        Updates camera synchronization for camera with serial ID cs.

        Args:
            cs (str): Camera serial ID number.
        """
        self.api_request(
            '/api/cameras/action',
            Serial=cs,
            Action='UPDATEMC',
        )
    
    def _connect_camera(self,cs):
        """_connect_camera

        Connects to camera with serial ID cs according to the specified configuration.

        Args:
            cs (str): Camera serial ID number.
        """
        self.api_request(
            'api/cameras/action',
            Serial=cs,
            Action='CONNECT',
            Iface='',
            Config=self.config,
            Codec=self.codec,
            Annotation=self.anno,
            Segtime=self.segtime,
        )
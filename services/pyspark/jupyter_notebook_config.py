# Configuration file for jupyter-server.
#
# This configuration increases limits to support large file uploads
# (e.g., audio files) in notebooks via ipywidgets.FileUpload

c = get_config()  # type: ignore

# ------------------------------------------------------------------------------
# ServerApp(JupyterApp) configuration
# ------------------------------------------------------------------------------

## Supply extra arguments that will be passed to Tornado's HTTPServer
#  Default: {}
c.ServerApp.tornado_settings = {
    "websocket_max_message_size": 500 * 1024 * 1024,  # 500MB (default: 10MB)
}

## (bytes/sec) Maximum rate at which stream output can be sent on iopub before
#  they are limited.
#  Default: 1000000
c.ServerApp.iopub_data_rate_limit = 100000000  # 100MB/s

## (msg/sec) Maximum rate at which messages can be sent on iopub before they are
#  limited.
#  Default: 1000
c.ServerApp.iopub_msg_rate_limit = 100000

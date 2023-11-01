from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from riglib import tablet_reward

hostName = "0.0.0.0" # open to anyone on the LAN - careful to not run this on the internet!
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        tablet_reward.open()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("<form action=\"\" method=\"post\"><button type=\"submit\" name=\"submit\" value=\"foobar\">Click me!</button></form>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def do_POST(self):
        tab = tablet_reward.open()
        tab.dispense()
        print('I got rest')

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
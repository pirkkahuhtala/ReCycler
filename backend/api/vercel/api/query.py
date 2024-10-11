from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from ..services import recycler_ai


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        query_param = query_params.get("query", [""])[0]

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        response_body = recycler_ai.query(query_param)
        self.wfile.write(response_body.encode("utf-8"))

        return

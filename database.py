"""MySQL event logging (reads on the main thread, writes on a background thread)."""
import queue
import threading
import time

import mysql.connector
from mysql.connector import Error


class Database:
    INSERT_SQL = (
        "INSERT INTO detection_logs "
        "(object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h) "
        "VALUES (%(object_class)s, %(confidence)s, %(bbox_x)s, %(bbox_y)s, %(bbox_w)s, %(bbox_h)s)"
    )

    def __init__(self, config: dict):
        self.config = config
        self.conn = None
        self.error = None
        self.connect()

    def connect(self):
        try:
            self.conn = mysql.connector.connect(connection_timeout=5, **self.config)
            self.error = None
            self._ensure_schema()
        except Error as e:
            self.conn, self.error = None, str(e)

    def _ensure_schema(self):
        """Creates the table on a fresh hosted database (the database itself must exist)."""
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS detection_logs ("
            "log_id INT AUTO_INCREMENT PRIMARY KEY, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
            "object_class VARCHAR(50) NOT NULL, confidence FLOAT NOT NULL, "
            "bbox_x INT NOT NULL, bbox_y INT NOT NULL, bbox_w INT NOT NULL, bbox_h INT NOT NULL)")
        self.conn.commit()
        cur.close()

    @property
    def ok(self) -> bool:
        try:
            return self.conn is not None and self.conn.is_connected()
        except Error:
            return False

    def insert(self, det: dict) -> bool:
        if not self.ok:
            return False
        try:
            cur = self.conn.cursor()
            cur.execute(self.INSERT_SQL, det)
            self.conn.commit()
            cur.close()
            return True
        except Error as e:
            self.error = str(e)
            return False

    def recent(self, limit: int = 10):
        if not self.ok:
            return []
        try:
            cur = self.conn.cursor(dictionary=True)
            cur.execute("SELECT log_id, timestamp, object_class, "
                        "ROUND(confidence*100, 1) AS confidence_pct, "
                        "bbox_x, bbox_y, bbox_w, bbox_h "
                        "FROM detection_logs ORDER BY log_id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            cur.close()
            return rows
        except Error as e:
            self.error = str(e)
            return []


class DBWriter:
    """Inserts happen in a background thread so database latency never slows the video."""

    def __init__(self, config: dict):
        self.db = Database(config)
        self.q = queue.Queue(maxsize=500)
        self.written = 0
        self._last_try = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, det: dict):
        try:
            self.q.put_nowait(det)
        except queue.Full:
            pass                                   # drop rather than lag the video

    def _run(self):
        while True:
            det = self.q.get()
            if not self.db.ok and time.time() - self._last_try > 5:
                self._last_try = time.time()
                self.db.connect()
            if self.db.insert(det):
                self.written += 1

import sqlite3
import os
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class IncidentDatabase:
    """
    SQLite persistence layer for industrial PPE compliance violations and zone intrusions.
    Provides fast indexing, auditing, status management, and KPI analytics.
    """

    def __init__(self, db_path: str = "data/database.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    worker_id INTEGER NOT NULL,
                    violation_type TEXT NOT NULL,
                    zone_name TEXT,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    details TEXT,
                    snapshot_path TEXT,
                    status TEXT DEFAULT 'NEW',
                    notes TEXT DEFAULT ''
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_timestamp 
                ON incidents(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_status 
                ON incidents(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_violation 
                ON incidents(violation_type)
            """)
            conn.commit()

    def log_incident(
        self,
        camera_id: str,
        worker_id: int,
        violation_type: str,
        severity: str = "HIGH",
        confidence: float = 0.85,
        zone_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Records a new violation incident and returns the incident ID.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        details_json = json.dumps(details or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    timestamp, camera_id, worker_id, violation_type,
                    zone_name, severity, confidence, details, snapshot_path, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW')
            """, (
                timestamp, camera_id, worker_id, violation_type,
                zone_name, severity, float(confidence), details_json, snapshot_path
            ))
            conn.commit()
            return cursor.lastrowid

    def get_incidents(
        self,
        limit: int = 100,
        offset: int = 0,
        violation_type: Optional[str] = None,
        zone_name: Optional[str] = None,
        status: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves filtered incidents ordered by timestamp descending.
        """
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []

        if violation_type and violation_type != "ALL":
            query += " AND violation_type = ?"
            params.append(violation_type)
        if zone_name and zone_name != "ALL":
            query += " AND zone_name = ?"
            params.append(zone_name)
        if status and status != "ALL":
            query += " AND status = ?"
            params.append(status)
        if camera_id and camera_id != "ALL":
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["details"] = json.loads(item["details"]) if item["details"] else {}
                except Exception:
                    pass
                results.append(item)
            return results

    def update_incident_status(self, incident_id: int, status: str, notes: Optional[str] = None) -> bool:
        """
        Updates the review/resolution status of an incident.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute(
                    "UPDATE incidents SET status = ?, notes = ? WHERE id = ?",
                    (status, notes, incident_id)
                )
            else:
                cursor.execute(
                    "UPDATE incidents SET status = ? WHERE id = ?",
                    (status, incident_id)
                )
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Computes aggregate metrics, violation distributions, and zone hazards for dashboard reporting.
        """
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total incidents
            cursor.execute(
                "SELECT COUNT(*) FROM incidents WHERE timestamp >= ?",
                (cutoff_time,)
            )
            total_incidents = cursor.fetchone()[0]

            # Pending/New
            cursor.execute(
                "SELECT COUNT(*) FROM incidents WHERE timestamp >= ? AND status = 'NEW'",
                (cutoff_time,)
            )
            pending_incidents = cursor.fetchone()[0]

            # Critical incidents
            cursor.execute(
                "SELECT COUNT(*) FROM incidents WHERE timestamp >= ? AND severity = 'CRITICAL'",
                (cutoff_time,)
            )
            critical_incidents = cursor.fetchone()[0]

            # Violation breakdown
            cursor.execute("""
                SELECT violation_type, COUNT(*) as count 
                FROM incidents 
                WHERE timestamp >= ? 
                GROUP BY violation_type 
                ORDER BY count DESC
            """, (cutoff_time,))
            violation_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

            # Zone breakdown
            cursor.execute("""
                SELECT COALESCE(zone_name, 'No Zone') as zone, COUNT(*) as count 
                FROM incidents 
                WHERE timestamp >= ? 
                GROUP BY zone 
                ORDER BY count DESC
            """, (cutoff_time,))
            zone_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

            # Hourly distribution
            cursor.execute("""
                SELECT substr(timestamp, 1, 13) as hour_bucket, COUNT(*) as count
                FROM incidents
                WHERE timestamp >= ?
                GROUP BY hour_bucket
                ORDER BY hour_bucket ASC
            """, (cutoff_time,))
            hourly_trend = [{"hour": row[0], "count": row[1]} for row in cursor.fetchall()]

            return {
                "total_incidents": total_incidents,
                "pending_incidents": pending_incidents,
                "critical_incidents": critical_incidents,
                "violation_breakdown": violation_breakdown,
                "zone_breakdown": zone_breakdown,
                "hourly_trend": hourly_trend
            }

    def export_to_csv(self, output_path: str = "data/incident_report.csv") -> str:
        """
        Exports the incident log to a CSV file for auditing.
        """
        import csv
        incidents = self.get_incidents(limit=10000)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not incidents:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "timestamp", "camera_id", "worker_id", "violation_type", "zone_name", "severity", "confidence", "status", "notes"])
            return output_path

        fieldnames = ["id", "timestamp", "camera_id", "worker_id", "violation_type", "zone_name", "severity", "confidence", "status", "notes", "snapshot_path"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for inc in incidents:
                writer.writerow(inc)

        return output_path

    def clear_all_incidents(self):
        """Clears all records (for testing or reset)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM incidents")
            conn.commit()

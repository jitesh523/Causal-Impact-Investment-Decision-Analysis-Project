"""
Alerts and Notifications Module
================================

Send alerts via Slack, Email, or webhooks when analysis results
meet specified conditions.

Author: Causal Impact Analysis Project
"""

import json
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import urllib.request
import urllib.error


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Supported alert channels."""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class Alert:
    """Represents an alert to be sent."""
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    name: str
    condition: Callable[[Dict], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    message_template: str
    cooldown_minutes: int = 60


class SlackNotifier:
    """Send notifications to Slack."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL (or set SLACK_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url:
            print("Warning: Slack webhook URL not configured")
            return False
        
        # Build Slack message
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.CRITICAL: "#f44336"
        }.get(alert.severity, "#808080")
        
        emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨"
        }.get(alert.severity, "📢")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{emoji} {alert.title}",
                "text": alert.message,
                "fields": [
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp, "short": True}
                ],
                "footer": "Causal Impact Analysis"
            }]
        }
        
        if alert.data:
            for key, value in list(alert.data.items())[:5]:
                payload["attachments"][0]["fields"].append({
                    "title": str(key),
                    "value": str(value),
                    "short": True
                })
        
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req)
            return True
        except urllib.error.URLError as e:
            print(f"Slack notification failed: {e}")
            return False


class EmailNotifier:
    """Send email notifications."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP port (default 587 for TLS)
            username: SMTP username
            password: SMTP password
            from_email: Sender email address
        """
        self.smtp_host = smtp_host or os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.username = username or os.environ.get('SMTP_USERNAME')
        self.password = password or os.environ.get('SMTP_PASSWORD')
        self.from_email = from_email or self.username
    
    def send(self, alert: Alert, to_emails: List[str]) -> bool:
        """Send alert via email."""
        if not all([self.smtp_host, self.username, self.password]):
            print("Warning: Email not configured")
            return False
        
        # Build email
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
<html>
<body>
<h2>{alert.title}</h2>
<p><strong>Severity:</strong> {alert.severity.value}</p>
<p><strong>Source:</strong> {alert.source}</p>
<p><strong>Time:</strong> {alert.timestamp}</p>
<hr>
<p>{alert.message}</p>
"""
        
        if alert.data:
            body += "<h3>Details:</h3><ul>"
            for key, value in alert.data.items():
                body += f"<li><strong>{key}:</strong> {value}</li>"
            body += "</ul>"
        
        body += "</body></html>"
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ', '.join(to_emails)
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, to_emails, msg.as_string())
            return True
        except Exception as e:
            print(f"Email notification failed: {e}")
            return False


class WebhookNotifier:
    """Send notifications to generic webhooks."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize webhook notifier.
        
        Args:
            webhook_url: Webhook endpoint URL
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        payload = {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "source": alert.source,
            "timestamp": alert.timestamp,
            "data": alert.data
        }
        
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers=self.headers
            )
            urllib.request.urlopen(req)
            return True
        except urllib.error.URLError as e:
            print(f"Webhook notification failed: {e}")
            return False


class AlertManager:
    """
    Central alert management system.
    
    Example:
        >>> manager = AlertManager()
        >>> manager.configure_slack(webhook_url="...")
        >>> manager.add_rule(AlertRule(
        ...     name="High ROI Alert",
        ...     condition=lambda d: d.get('roi', 0) > 500,
        ...     severity=AlertSeverity.INFO,
        ...     channels=[AlertChannel.SLACK],
        ...     message_template="ROI exceeded 500%: {roi}%"
        ... ))
        >>> manager.check_and_alert({'roi': 743})
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self._slack: Optional[SlackNotifier] = None
        self._email: Optional[EmailNotifier] = None
        self._webhooks: Dict[str, WebhookNotifier] = {}
        self._rules: List[AlertRule] = []
        self._alert_history: List[Dict] = []
        self._last_alert_times: Dict[str, datetime] = {}
    
    def configure_slack(self, webhook_url: str):
        """Configure Slack notifications."""
        self._slack = SlackNotifier(webhook_url)
    
    def configure_email(self, **kwargs):
        """Configure email notifications."""
        self._email = EmailNotifier(**kwargs)
    
    def add_webhook(self, name: str, url: str, headers: Optional[Dict] = None):
        """Add a webhook endpoint."""
        self._webhooks[name] = WebhookNotifier(url, headers)
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self._rules.append(rule)
    
    def _should_alert(self, rule: AlertRule) -> bool:
        """Check if alert should be sent (respecting cooldown)."""
        last_time = self._last_alert_times.get(rule.name)
        if not last_time:
            return True
        
        elapsed = (datetime.now() - last_time).total_seconds() / 60
        return elapsed >= rule.cooldown_minutes
    
    def check_and_alert(
        self,
        data: Dict[str, Any],
        source: str = "analysis",
        email_recipients: Optional[List[str]] = None
    ) -> List[Alert]:
        """
        Check rules and send alerts for matching conditions.
        
        Args:
            data: Data to check against rules
            source: Source identifier
            email_recipients: Email addresses for email alerts
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self._rules:
            try:
                if rule.condition(data) and self._should_alert(rule):
                    # Format message
                    message = rule.message_template.format(**data)
                    
                    alert = Alert(
                        title=rule.name,
                        message=message,
                        severity=rule.severity,
                        source=source,
                        timestamp=datetime.now().isoformat(),
                        data=data
                    )
                    
                    # Send to configured channels
                    for channel in rule.channels:
                        self._send_alert(alert, channel, email_recipients)
                    
                    triggered.append(alert)
                    self._last_alert_times[rule.name] = datetime.now()
                    self._alert_history.append(asdict(alert))
                    
            except Exception as e:
                print(f"Rule {rule.name} failed: {e}")
        
        return triggered
    
    def _send_alert(
        self,
        alert: Alert,
        channel: AlertChannel,
        email_recipients: Optional[List[str]] = None
    ):
        """Send alert to specific channel."""
        if channel == AlertChannel.SLACK and self._slack:
            self._slack.send(alert)
        
        elif channel == AlertChannel.EMAIL and self._email and email_recipients:
            self._email.send(alert, email_recipients)
        
        elif channel == AlertChannel.WEBHOOK:
            for webhook in self._webhooks.values():
                webhook.send(alert)
        
        elif channel == AlertChannel.CONSOLE:
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.CRITICAL: "🚨"
            }.get(alert.severity, "📢")
            print(f"\n{severity_emoji} ALERT: {alert.title}")
            print(f"   {alert.message}")
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "manual",
        data: Optional[Dict] = None,
        channels: Optional[List[AlertChannel]] = None,
        email_recipients: Optional[List[str]] = None
    ):
        """Send a manual alert."""
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        
        channels = channels or [AlertChannel.CONSOLE]
        for channel in channels:
            self._send_alert(alert, channel, email_recipients)
        
        self._alert_history.append(asdict(alert))
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history."""
        return self._alert_history[-limit:]


# Pre-configured rules for common scenarios
def get_default_rules() -> List[AlertRule]:
    """Get default alert rules for causal analysis."""
    return [
        AlertRule(
            name="Significant Effect Detected",
            condition=lambda d: d.get('p_value', 1) < 0.05,
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.CONSOLE],
            message_template="Analysis found significant effect (p={p_value:.4f})"
        ),
        AlertRule(
            name="High ROI Alert",
            condition=lambda d: d.get('roi', 0) > 500,
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.CONSOLE],
            message_template="Exceptional ROI detected: {roi:.1f}%"
        ),
        AlertRule(
            name="Negative Effect Warning",
            condition=lambda d: d.get('effect', 0) < 0 and d.get('p_value', 1) < 0.05,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE],
            message_template="Negative significant effect: {effect:.2f}"
        ),
        AlertRule(
            name="Drift Detected",
            condition=lambda d: d.get('drift_detected', False),
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE],
            message_template="Model drift detected! {drift_message}"
        )
    ]


def main():
    """Demo alerts module."""
    print("=" * 60)
    print("ALERTS & NOTIFICATIONS DEMO")
    print("=" * 60)
    
    # Create alert manager
    manager = AlertManager()
    
    # Add default rules
    for rule in get_default_rules():
        manager.add_rule(rule)
    
    # Simulate analysis results
    print("\n1. Testing with positive results:")
    results1 = {
        'effect': 42137.64,
        'p_value': 0.001,
        'roi': 743.0
    }
    alerts = manager.check_and_alert(results1, source='campaign_analysis')
    print(f"   Triggered {len(alerts)} alert(s)")
    
    print("\n2. Testing with negative results:")
    results2 = {
        'effect': -5000,
        'p_value': 0.02,
        'roi': -25
    }
    alerts = manager.check_and_alert(results2, source='campaign_analysis')
    print(f"   Triggered {len(alerts)} alert(s)")
    
    print("\n3. Testing drift alert:")
    results3 = {
        'drift_detected': True,
        'drift_message': 'Feature distribution changed significantly'
    }
    alerts = manager.check_and_alert(results3, source='monitoring')
    print(f"   Triggered {len(alerts)} alert(s)")
    
    print("\n4. Manual alert:")
    manager.send_alert(
        title="Analysis Complete",
        message="Weekly analysis finished successfully",
        severity=AlertSeverity.INFO
    )
    
    print(f"\n✓ Alert demo completed! Total alerts: {len(manager.get_history())}")


if __name__ == '__main__':
    main()

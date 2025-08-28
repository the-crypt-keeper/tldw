# email_service.py
# Description: Email service with mock provider for development and testing
#
# Imports
import os
import smtplib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
#
# 3rd-party imports
from jinja2 import Template
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.exceptions import ExternalServiceError

#######################################################################################################################
#
# Email Templates
#

EMAIL_TEMPLATES = {
    "password_reset": {
        "subject": "Password Reset Request - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #007bff; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #007bff; 
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; 
                  font-size: 0.9em; color: #6c757d; }
        .warning { background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; 
                   margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
        </div>
        <div class="content">
            <h2>Password Reset Request</h2>
            <p>Hello {{ username }},</p>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <center>
                <a href="{{ reset_link }}" class="button">Reset Password</a>
            </center>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; background: #fff; padding: 10px; border: 1px solid #dee2e6;">
                {{ reset_link }}
            </p>
            <div class="warning">
                <strong>⚠️ Security Notice:</strong><br>
                This link will expire in {{ expiry_hours }} hour(s).<br>
                If you didn't request this, please ignore this email or contact support if you're concerned.
            </div>
        </div>
        <div class="footer">
            <p>This is an automated message from {{ app_name }}.</p>
            <p>Request made from IP: {{ ip_address }}<br>
            Time: {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Password Reset Request - {{ app_name }}

Hello {{ username }},

We received a request to reset your password.

To reset your password, visit this link:
{{ reset_link }}

This link will expire in {{ expiry_hours }} hour(s).

If you didn't request this, please ignore this email.

Security Information:
- Request from IP: {{ ip_address }}
- Time: {{ timestamp }}

This is an automated message from {{ app_name }}.
"""
    },
    "email_verification": {
        "subject": "Verify Your Email - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #28a745; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #28a745; 
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; 
                  font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to {{ app_name }}!</h1>
        </div>
        <div class="content">
            <h2>Verify Your Email Address</h2>
            <p>Hello {{ username }},</p>
            <p>Thank you for registering! Please verify your email address to activate your account:</p>
            <center>
                <a href="{{ verification_link }}" class="button">Verify Email</a>
            </center>
            <p>Or copy and paste this link:</p>
            <p style="word-break: break-all; background: #fff; padding: 10px; border: 1px solid #dee2e6;">
                {{ verification_link }}
            </p>
            <p><strong>This link expires in {{ expiry_hours }} hours.</strong></p>
        </div>
        <div class="footer">
            <p>If you didn't create an account, please ignore this email.</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Welcome to {{ app_name }}!

Hello {{ username }},

Thank you for registering! Please verify your email address by visiting:

{{ verification_link }}

This link expires in {{ expiry_hours }} hours.

If you didn't create an account, please ignore this email.
"""
    },
    "mfa_enabled": {
        "subject": "Two-Factor Authentication Enabled - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #17a2b8; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .code-box { background: #fff; border: 2px solid #17a2b8; padding: 15px; 
                    margin: 20px 0; border-radius: 5px; font-family: monospace; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; 
                  font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Two-Factor Authentication Enabled</h1>
        </div>
        <div class="content">
            <h2>Your Account is Now More Secure!</h2>
            <p>Hello {{ username }},</p>
            <p>Two-factor authentication has been successfully enabled on your account.</p>
            
            <h3>Your Backup Codes</h3>
            <p>Save these backup codes in a safe place. Each code can be used once if you lose access to your authenticator app:</p>
            <div class="code-box">
                {% for code in backup_codes %}
                {{ code }}<br>
                {% endfor %}
            </div>
            
            <p><strong>⚠️ Important:</strong> Store these codes securely. You won't be able to see them again.</p>
        </div>
        <div class="footer">
            <p>Enabled from IP: {{ ip_address }}<br>
            Time: {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Two-Factor Authentication Enabled - {{ app_name }}

Hello {{ username }},

Two-factor authentication has been successfully enabled on your account.

Your Backup Codes (save these securely):
{% for code in backup_codes %}
{{ code }}
{% endfor %}

These codes can be used once each if you lose access to your authenticator app.

Enabled from IP: {{ ip_address }}
Time: {{ timestamp }}
"""
    }
}

#######################################################################################################################
#
# Email Service Class
#

class EmailService:
    """
    Email service with support for multiple providers including mock for development
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize email service"""
        self.settings = settings or get_settings()
        
        # Email configuration
        self.provider = os.getenv("EMAIL_PROVIDER", "mock")  # mock, smtp, sendgrid, etc.
        self.mock_output = os.getenv("EMAIL_MOCK_OUTPUT", "console")  # console, file, both
        self.mock_file_path = Path(os.getenv("EMAIL_MOCK_FILE_PATH", "./mock_emails"))
        
        # SMTP configuration (if using SMTP)
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        
        # Default sender
        self.default_sender = os.getenv("EMAIL_FROM", "noreply@example.com")
        self.app_name = os.getenv("APP_NAME", "TLDW Server")
        
        # Create mock email directory if needed
        if self.provider == "mock" and self.mock_output in ["file", "both"]:
            self.mock_file_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EmailService initialized with provider: {self.provider}")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        from_email: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Send an email
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML version of email
            text_body: Plain text version (optional)
            from_email: Sender email (uses default if not provided)
            attachments: List of attachments
            
        Returns:
            True if email was sent successfully
        """
        from_email = from_email or self.default_sender
        
        if self.provider == "mock":
            return await self._send_mock_email(
                to_email, subject, html_body, text_body, from_email, attachments
            )
        elif self.provider == "smtp":
            return await self._send_smtp_email(
                to_email, subject, html_body, text_body, from_email, attachments
            )
        else:
            logger.error(f"Unsupported email provider: {self.provider}")
            return False
    
    async def _send_mock_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str],
        from_email: str,
        attachments: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Send mock email for development/testing"""
        
        timestamp = datetime.utcnow().isoformat()
        email_id = f"{timestamp}_{to_email.replace('@', '_at_')}"
        
        # Create email data structure
        email_data = {
            "id": email_id,
            "timestamp": timestamp,
            "from": from_email,
            "to": to_email,
            "subject": subject,
            "html_body": html_body,
            "text_body": text_body or "",
            "attachments": len(attachments) if attachments else 0,
            "provider": "mock"
        }
        
        # Output to console
        if self.mock_output in ["console", "both"]:
            logger.info("=" * 80)
            logger.info("📧 MOCK EMAIL SENT")
            logger.info("=" * 80)
            logger.info(f"From: {from_email}")
            logger.info(f"To: {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info("-" * 80)
            if text_body:
                logger.info("Text Body:")
                logger.info(text_body[:500] + ("..." if len(text_body) > 500 else ""))
            logger.info("-" * 80)
            logger.info(f"HTML Body Length: {len(html_body)} characters")
            if attachments:
                logger.info(f"Attachments: {len(attachments)}")
            logger.info("=" * 80)
        
        # Save to file
        if self.mock_output in ["file", "both"]:
            file_path = self.mock_file_path / f"{email_id}.json"
            with open(file_path, "w") as f:
                json.dump(email_data, f, indent=2)
            
            # Also save HTML for viewing
            html_path = self.mock_file_path / f"{email_id}.html"
            with open(html_path, "w") as f:
                f.write(html_body)
            
            logger.debug(f"Mock email saved to: {file_path}")
        
        # Simulate small delay
        await asyncio.sleep(0.1)
        
        return True
    
    async def _send_smtp_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str],
        from_email: str,
        attachments: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Send email via SMTP"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email
            
            # Add text and HTML parts
            if text_body:
                text_part = MIMEText(text_body, 'plain')
                msg.attach(text_part)
            
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False
    
    async def send_password_reset_email(
        self,
        to_email: str,
        username: str,
        reset_token: str,
        ip_address: str = "Unknown",
        base_url: Optional[str] = None
    ) -> bool:
        """
        Send password reset email
        
        Args:
            to_email: Recipient email
            username: User's username
            reset_token: Password reset token
            ip_address: IP address of request
            base_url: Base URL for reset link
            
        Returns:
            True if sent successfully
        """
        base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")
        reset_link = f"{base_url}/auth/reset-password?token={reset_token}"
        
        template_data = {
            "app_name": self.app_name,
            "username": username,
            "reset_link": reset_link,
            "expiry_hours": 1,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Render templates
        html_template = Template(EMAIL_TEMPLATES["password_reset"]["html"])
        text_template = Template(EMAIL_TEMPLATES["password_reset"]["text"])
        
        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["password_reset"]["subject"]).render(**template_data)
        
        return await self.send_email(to_email, subject, html_body, text_body)
    
    async def send_verification_email(
        self,
        to_email: str,
        username: str,
        verification_token: str,
        base_url: Optional[str] = None
    ) -> bool:
        """Send email verification email"""
        
        base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")
        verification_link = f"{base_url}/auth/verify-email?token={verification_token}"
        
        template_data = {
            "app_name": self.app_name,
            "username": username,
            "verification_link": verification_link,
            "expiry_hours": 24
        }
        
        # Render templates
        html_template = Template(EMAIL_TEMPLATES["email_verification"]["html"])
        text_template = Template(EMAIL_TEMPLATES["email_verification"]["text"])
        
        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["email_verification"]["subject"]).render(**template_data)
        
        return await self.send_email(to_email, subject, html_body, text_body)
    
    async def send_mfa_enabled_email(
        self,
        to_email: str,
        username: str,
        backup_codes: List[str],
        ip_address: str = "Unknown"
    ) -> bool:
        """Send MFA enabled notification with backup codes"""
        
        template_data = {
            "app_name": self.app_name,
            "username": username,
            "backup_codes": backup_codes,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Render templates
        html_template = Template(EMAIL_TEMPLATES["mfa_enabled"]["html"])
        text_template = Template(EMAIL_TEMPLATES["mfa_enabled"]["text"])
        
        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["mfa_enabled"]["subject"]).render(**template_data)
        
        return await self.send_email(to_email, subject, html_body, text_body)


#######################################################################################################################
#
# Module Functions
#

# Global instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get email service singleton instance"""
    global _email_service
    if not _email_service:
        _email_service = EmailService()
    return _email_service


#
# End of email_service.py
#######################################################################################################################
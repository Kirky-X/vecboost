## MODIFIED Requirements

### Requirement: JWT Secret Configuration

The system SHALL require JWT secret to be set via environment variable `VECBOOST_JWT_SECRET` for production deployments.

The system SHALL validate that the JWT secret meets minimum entropy requirements:
- Minimum length of 32 characters
- Minimum entropy of 128 bits

#### Scenario: JWT secret from environment variable
- **WHEN** environment variable `VECBOOST_JWT_SECRET` is set
- **THEN** the system SHALL use this value for JWT operations

#### Scenario: JWT secret validation failure
- **WHEN** JWT secret is less than 32 characters
- **THEN** the system SHALL refuse to start with a configuration error

#### Scenario: JWT secret not provided
- **WHEN** environment variable is not set and config file value is empty
- **THEN** the system SHALL refuse to start with a clear error message

---

### Requirement: Default Admin Password

The system SHALL generate a secure random password on first startup if no password is provided.

The default password SHALL meet the following requirements:
- Minimum 16 characters
- Contains uppercase and lowercase letters
- Contains numbers
- Contains special characters

#### Scenario: No password provided
- **WHEN** `VECBOOST_ADMIN_PASSWORD` environment variable is not set
- **THEN** the system SHALL generate a random secure password

#### Scenario: Password from environment variable
- **WHEN** `VECBOOST_ADMIN_PASSWORD` is set
- **THEN** the system SHALL use this password directly

#### Scenario: Weak password rejected
- **WHEN** provided password does not meet complexity requirements
- **THEN** the system SHALL reject the password and require a valid one

---

### Requirement: Token Blacklist Storage

The system SHALL support multiple storage backends for token blacklists.

The system SHALL support:
- In-memory storage for development and testing
- Redis storage for production deployments with multiple instances

#### Scenario: Memory blacklist storage
- **WHEN** Redis is not configured or unavailable
- **THEN** the system SHALL use in-memory token blacklist

#### Scenario: Redis blacklist storage
- **WHEN** Redis connection is configured and available
- **THEN** the system SHALL use Redis for token blacklist storage

#### Scenario: Redis connection failure
- **WHEN** Redis connection fails
- **THEN** the system SHALL fall back to in-memory storage
- **AND** the system SHALL log a warning message

---

### Requirement: CSRF Token Storage

The system SHALL support multiple storage backends for CSRF tokens.

The system SHALL support:
- In-memory storage for development and testing
- Redis storage for production deployments

#### Scenario: Memory CSRF storage
- **WHEN** Redis is not configured
- **THEN** the system SHALL use in-memory CSRF token storage

#### Scenario: Redis CSRF storage
- **WHEN** Redis is configured and available
- **THEN** the system SHALL use Redis for CSRF token storage

---

## ADDED Requirements

### Requirement: Secure File Permissions

The system SHALL set restrictive permissions on files containing sensitive data.

For files containing encryption keys or sensitive data:
- On Unix-like systems, permissions SHALL be set to `0o600` (owner read/write only)
- The file SHALL NOT be readable by group or others

#### Scenario: Creating sensitive file
- **WHEN** the system creates a file for storing sensitive data
- **THEN** the system SHALL set file permissions to `0o600`
- **AND** the system SHALL verify the permissions were set correctly

---

### Requirement: Configuration Logging

The system SHALL NOT log sensitive configuration values.

The following fields SHALL NOT be logged:
- `auth.jwt_secret`
- `auth.default_admin_password`
- `security.encryption_key`

#### Scenario: Logging configuration
- **WHEN** the system logs configuration information
- **THEN** sensitive fields SHALL be redacted or omitted
- **AND** the log SHALL indicate whether sensitive fields are present without revealing values

---

### Requirement: Audit Log Context

The system SHALL include comprehensive context in audit log entries.

Each audit log entry SHALL include:
- User agent string
- Request path and method
- Request ID for correlation
- Session ID when available
- Geo location when available
- Timestamp in ISO 8601 format

#### Scenario: Audit log with full context
- **WHEN** an auditable event occurs
- **THEN** the audit log entry SHALL include all required context fields
- **AND** fields that are not available SHALL be set to `null`

#### Scenario: Login success audit
- **WHEN** a user successfully authenticates
- **THEN** the audit log SHALL include:
  - Username
  - IP address
  - User agent
  - Request path
  - Session ID (if available)

#### Scenario: Login failure audit
- **WHEN** a user fails to authenticate
- **THEN** the audit log SHALL include:
  - Attempted username
  - IP address
  - User agent
  - Failure reason (without sensitive data)

# Security Policy

## Supported Versions

We actively support the following versions of OpenGrid with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | ✅ Yes             |
| 0.2.x   | ✅ Yes             |
| < 0.2   | ❌ No              |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in OpenGrid, please follow these steps:

### 1. Do Not Open a Public Issue

Please **do not** open a public GitHub issue for security vulnerabilities.

### 2. Report Privately

Send an email to **nikjois@llamasearch.ai** with:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if available)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: Varies based on complexity
- **Public Disclosure**: After fix is available

### 4. Responsible Disclosure

We follow responsible disclosure practices:

1. We will acknowledge receipt of your report
2. We will investigate and validate the vulnerability
3. We will develop and test a fix
4. We will coordinate disclosure timing with you
5. We will credit you in the security advisory (if desired)

## Security Best Practices

### For Users

- Keep OpenGrid updated to the latest version
- Use environment variables for sensitive configuration
- Never commit API keys or passwords to version control
- Run OpenGrid in isolated environments when possible
- Review and validate all network models before analysis

### For Developers

- Follow secure coding practices
- Validate all user inputs
- Use parameterized queries for database operations
- Implement proper authentication and authorization
- Regular dependency updates and vulnerability scanning

## Security Features

OpenGrid includes several security features:

- **Input Validation**: All user inputs are validated
- **API Rate Limiting**: Protection against abuse
- **Secure Defaults**: Conservative default configurations
- **No Persistent Storage**: No sensitive data stored by default
- **Sandboxed Execution**: Analysis runs in isolated environments

## Dependencies

We regularly monitor and update dependencies for security vulnerabilities. You can check the current dependency security status by running:

```bash
pip-audit
```

## Bug Bounty

Currently, we do not have a formal bug bounty program, but we appreciate security researchers who report vulnerabilities responsibly.

## Contact

For any security-related questions or concerns:

- **Email**: nikjois@llamasearch.ai
- **Response Time**: Within 48 hours
- **PGP Key**: Available upon request

---

Thank you for helping keep OpenGrid secure! 
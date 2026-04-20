---
name: doc
description: Documentation generation and update. Creates README, API docs, architecture docs. Use when documentation is needed or outdated.
---

# Documentation Skill

## Usage
```
/doc [type] [target]
```

### Parameters
- `type`: readme | api | arch | guide | changelog
- `target`: File or directory to document

### Examples
```bash
/doc readme
/doc api src/routes/
/doc arch
/doc guide deployment
/doc changelog v1.2.0
```

## Documentation Types

### README (`/doc readme`)
Creates or updates project README.md:

```markdown
# Project Name

Brief description.

## Features
- Feature 1
- Feature 2

## Installation
```bash
npm install project-name
```

## Usage
```javascript
import { feature } from 'project-name';
```

## API Reference
See [API Documentation](docs/api.md)

## Contributing
See [Contributing Guide](CONTRIBUTING.md)

## License
MIT
```

### API Documentation (`/doc api`)
Documents API endpoints or module exports:

```markdown
# API Reference

## Endpoints

### `GET /api/users`
Returns list of users.

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| limit | number | No | Max results |

**Response**
```json
{
  "data": [{ "id": 1, "name": "John" }],
  "total": 100
}
```

### `POST /api/users`
Creates a new user.

**Body**
```json
{
  "name": "string",
  "email": "string"
}
```
```

### Architecture (`/doc arch`)
Documents system architecture:

```markdown
# Architecture

## System Overview
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Client  │────▶│   API   │────▶│   DB    │
└─────────┘     └─────────┘     └─────────┘
```

## Components

### Frontend
- Framework: React
- State: Redux
- Routing: React Router

### Backend
- Runtime: Node.js
- Framework: Express
- Database: PostgreSQL

## Data Flow
1. User action triggers API call
2. API validates and processes
3. Database query/update
4. Response returned

## Directory Structure
```
src/
├── components/    # UI components
├── services/      # Business logic
├── models/        # Data models
└── utils/         # Utilities
```
```

### Guide (`/doc guide`)
Creates how-to guides:

```markdown
# {Topic} Guide

## Overview
What this guide covers.

## Prerequisites
- Requirement 1
- Requirement 2

## Steps

### Step 1: Setup
Description and code.

### Step 2: Configuration
Description and code.

### Step 3: Execution
Description and code.

## Troubleshooting

### Common Issue 1
Solution.

### Common Issue 2
Solution.

## Related
- [Other Guide](other.md)
```

### Changelog (`/doc changelog`)
Updates CHANGELOG.md:

```markdown
# Changelog

## [1.2.0] - YYYY-MM-DD

### Added
- New feature X
- New feature Y

### Changed
- Improved performance of Z

### Fixed
- Bug in component A
- Issue with feature B

### Deprecated
- Old method C (use D instead)

### Removed
- Legacy feature E

### Security
- Fixed vulnerability in F
```

## Workflow

### 1. Analyze Target
```bash
# Find existing docs
Glob: **/*.md, docs/**

# Find code to document
Glob: src/**/*.{ts,js}
Grep: "export|module\.exports"
```

### 2. Extract Information
- Function signatures
- Class definitions
- Type definitions
- Comments and JSDoc

### 3. Generate Documentation
- Use consistent formatting
- Include examples
- Add cross-references

### 4. Validate
- Check for broken links
- Verify code examples
- Ensure completeness

## Best Practices

1. **Be Concise**: Direct language, no fluff
2. **Use Examples**: Code > description
3. **Stay Current**: Update with code changes
4. **Structure Well**: Consistent headers
5. **Link Related**: Cross-reference docs

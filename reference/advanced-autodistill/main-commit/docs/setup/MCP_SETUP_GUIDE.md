# Model Control Panel (MCP) Setup Guide

This guide provides detailed instructions for setting up and running the Model Control Panel (MCP) for the Autodistill project.

## System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: Version 3.8 or higher
- **Node.js**: Version 14 or higher
- **npm**: Version 6 or higher
- **Git**: For version control
- **Storage**: At least 1GB free disk space
- **Memory**: Minimum 4GB RAM recommended

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project-agi
```

### 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\\Scripts\\activate
```

### 3. Install Backend Dependencies

```bash
cd mcp
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your configuration
# Example minimal configuration:
MCP_SECRET_KEY="your-secret-key-here"
MCP_DATABASE_URL="sqlite:///./mcp.db"
```

### 5. Initialize the Database

```bash
# Run migrations
cd migrations
alembic upgrade head
cd ..
```

### 6. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Running the Application

There are several ways to run the application, depending on your needs:

### Option 1: Development Mode (Separate Backend and Frontend)

This option is best for development as it provides hot-reloading for both frontend and backend.

#### Terminal 1 - Run Backend:

```bash
cd project-agi/mcp
source ../venv/bin/activate  # If using virtual environment
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 - Run Frontend:

```bash
cd project-agi/mcp/frontend
npm start
```

- Backend will be available at: http://localhost:8000
- Frontend will be available at: http://localhost:3000
- API documentation will be available at: http://localhost:8000/docs

### Option 2: Using the Run Script

The project includes a convenient script that can handle various startup scenarios:

```bash
cd project-agi/mcp
chmod +x run_mcp.sh  # Make the script executable (first time only)
./run_mcp.sh --dev --frontend
```

This will:
- Start the backend with auto-reload on port 8000
- Start the frontend development server on port 3000

For more options:

```bash
./run_mcp.sh --help
```

### Option 3: Production Mode

For a production-like environment where the frontend is built and served by the backend:

```bash
cd project-agi/mcp
./run_mcp.sh --build
```

This will:
1. Build the React frontend
2. Copy the built files to the backend's static directory
3. Start the backend server in production mode
4. The application will be available at: http://localhost:8000

## Accessing the Application

Once the application is running:

1. Open your browser and go to:
   - Development mode: http://localhost:3000
   - Production mode: http://localhost:8000

2. Log in with the default credentials:
   - Username: `admin`
   - Password: `adminpassword`

## Verifying Installation

To verify that everything is set up correctly:

```bash
cd project-agi/mcp
python src/utils/api_test.py
```

This will run basic tests against the API to ensure everything is functioning properly.

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```
   ERROR: Address already in use
   ```
   Solution: Find the process using the port and kill it, or use a different port with `--port`.

2. **Database migration errors**:
   ```
   ERROR: Can't locate revision identified by '...'
   ```
   Solution: Delete the `mcp.db` file and rerun migrations.

3. **Frontend build errors**:
   ```
   Error: Cannot find module '...'
   ```
   Solution: Make sure you've run `npm install` in the frontend directory.

4. **Authentication issues**:
   ```
   {"detail":"Could not validate credentials"}
   ```
   Solution: Check that your `.env` file has the correct `MCP_SECRET_KEY`.

5. **CORS errors in browser console**:
   Solution: Make sure you're running the backend on the expected host/port.

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for specific error messages
2. Review the README.md file for additional information
3. Contact the project maintainers

## Next Steps

After successful setup, you can:

1. Create categories by navigating to the Categories page
2. Configure AI models in the Models page
3. Set up and run a pipeline in the Pipeline page
4. Check the Dashboard for an overview of the system status

## Updating the Application

To update the application to the latest version:

```bash
git pull
cd mcp
pip install -r requirements.txt
cd migrations
alembic upgrade head
cd ../frontend
npm install
```

Then restart the application using your preferred method. 
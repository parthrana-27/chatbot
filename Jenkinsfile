pipeline {
    agent any

    options {
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        APP_NAME = 'chatbot-app'
        IMAGE_NAME = 'chatbot-app'
        IMAGE_TAG = "${BUILD_NUMBER}"
        // HUGGINGFACEHUB_API_TOKEN = credentials('HUGGINGFACEHUB_API_TOKEN') // Optional: configured in Jenkins Credentials
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo "=========================================="
                echo "Stage 1: Checking out repository code..."
                echo "=========================================="
                checkout scm
            }
        }

        stage('Setup Python & Dependencies') {
            steps {
                echo "=========================================="
                echo "Stage 2: Setting up virtual environment..."
                echo "=========================================="
                sh '''
                    python3 -m venv venv || python -m venv venv
                    . venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install flake8 pytest pytest-cov
                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    fi
                '''
            }
        }

        stage('Lint & Code Analysis') {
            steps {
                echo "=========================================="
                echo "Stage 3: Running Flake8 Lint Checks..."
                echo "=========================================="
                sh '''
                    . venv/bin/activate
                    # Stop build if there are Python syntax errors or undefined symbols
                    flake8 backend/ app.py run.py --count --select=E9,F63,F7,F82 --show-source --statistics
                    # Exit-zero treats all warnings as non-fatal
                    flake8 backend/ app.py run.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
                '''
            }
        }

        stage('Run Unit Tests') {
            steps {
                echo "=========================================="
                echo "Stage 4: Running Unit Tests & Coverage..."
                echo "=========================================="
                sh '''
                    . venv/bin/activate
                    mkdir -p test-reports
                    HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:-hf_dummy_ci_token}" \
                    pytest --junitxml=test-reports/results.xml tests/
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "=========================================="
                echo "Stage 5: Building Docker Image..."
                echo "=========================================="
                script {
                    def dockerExists = sh(script: 'command -v docker', returnStatus: true) == 0
                    if (dockerExists) {
                        sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
                        echo "Docker image ${IMAGE_NAME}:${IMAGE_TAG} built successfully."
                    } else {
                        echo "WARNING: Docker CLI is not installed or accessible on this agent node. Skipping Docker build stage."
                    }
                }
            }
        }

        stage('Deploy / Run Verification (Optional)') {
            steps {
                echo "=========================================="
                echo "Stage 6: Verifying container image setup..."
                echo "=========================================="
                script {
                    def dockerExists = sh(script: 'command -v docker', returnStatus: true) == 0
                    if (dockerExists) {
                        sh "docker images | grep ${IMAGE_NAME} || true"
                    } else {
                        echo "Skipping container deployment verification."
                    }
                }
            }
        }
    }

    post {
        always {
            echo "Pipeline finished. Cleaning up temporary artifacts..."
            junit allowEmptyResults: true, testResults: 'test-reports/*.xml'
        }
        success {
            echo "SUCCESS: Jenkins CI/CD Pipeline completed successfully for build #${BUILD_NUMBER}!"
        }
        failure {
            echo "FAILURE: Jenkins CI/CD Pipeline failed for build #${BUILD_NUMBER}. Please check build logs."
        }
    }
}

// Exoplanet Explorer Landing Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const startButton = document.getElementById('startExploring');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements for scroll animations
    document.querySelectorAll('.feature-card, .about-content, .about-image').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Handle Start Exploring button click
    if (startButton) {
        startButton.addEventListener('click', function() {
            // Show loading overlay
            loadingOverlay.classList.add('show');
            
            // Check if Streamlit is already running
            checkStreamlitStatus()
                .then(isRunning => {
                    if (isRunning) {
                        // Streamlit is running, redirect to it
                        window.location.href = 'http://localhost:8501';
                    } else {
                        // Start Streamlit application
                        startStreamlitApp();
                    }
                })
                .catch(error => {
                    console.error('Error checking Streamlit status:', error);
                    // Try to start Streamlit anyway
                    startStreamlitApp();
                });
        });
    }
    
    // Parallax effect for background
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const background = document.querySelector('.background');
        if (background) {
            background.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });
    
    // Dynamic text effect
    function createTextEffect() {
        const gradientText = document.querySelector('.gradient-text');
        if (gradientText) {
            gradientText.addEventListener('mouseenter', function() {
                this.style.animation = 'gradient-flow 0.5s ease-in-out';
            });
            
            gradientText.addEventListener('mouseleave', function() {
                this.style.animation = 'gradient-flow 3s ease-in-out infinite alternate';
            });
        }
    }
    
    createTextEffect();
});

// Function to check if Streamlit is running
async function checkStreamlitStatus() {
    try {
        const response = await fetch('http://localhost:8501', { 
            method: 'GET',
            mode: 'no-cors' // This will help avoid CORS issues
        });
        return true; // If we get here, Streamlit is running
    } catch (error) {
        return false; // Streamlit is not running
    }
}

// Function to start Streamlit application
function startStreamlitApp() {
    // Show loading message
    updateLoadingText('Starting Streamlit server...', 'This may take a few moments...');
    
    // Method 1: Try to use a backend service to start Streamlit
    fetch('/start-streamlit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        }
        throw new Error('Backend not available');
    })
    .then(data => {
        if (data.success) {
            updateLoadingText('Server started successfully!', 'Redirecting to the application...');
            // Wait a moment for Streamlit to fully start
            setTimeout(() => {
                window.location.href = 'http://localhost:8501';
            }, 2000);
        } else {
            throw new Error('Failed to start server');
        }
    })
    .catch(error => {
        console.log('Backend method failed, trying alternative approach:', error);
        // Method 2: Show instructions to user
        showStreamlitInstructions();
    });
}

// Function to show instructions when automatic start fails
function showStreamlitInstructions() {
    const loadingContent = document.querySelector('.loading-content');
    if (loadingContent) {
        loadingContent.innerHTML = `
            <div class="loading-spinner" style="display: none;"></div>
            <h3 class="loading-title">Manual Setup Required</h3>
            <div style="text-align: left; max-width: 500px; margin: 0 auto;">
                <p class="loading-text" style="margin-bottom: 1rem;">
                    To start the Exoplanet Explorer, please run the following command in your terminal:
                </p>
                <div style="background: rgba(45, 212, 191, 0.1); border: 1px solid rgba(45, 212, 191, 0.3); border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; font-family: monospace; font-size: 0.875rem;">
                    streamlit run app.py
                </div>
                <p class="loading-text" style="margin-bottom: 1.5rem;">
                    Then visit <strong>http://localhost:8501</strong> in your browser.
                </p>
                <div style="display: flex; gap: 1rem; justify-content: center;">
                    <button onclick="closeLoading()" style="padding: 0.75rem 1.5rem; background: var(--primary); color: var(--primary-foreground); border: none; border-radius: 0.5rem; cursor: pointer; font-family: inherit;">
                        Got it
                    </button>
                    <button onclick="tryRedirect()" style="padding: 0.75rem 1.5rem; background: transparent; color: var(--primary); border: 1px solid var(--primary); border-radius: 0.5rem; cursor: pointer; font-family: inherit;">
                        Try localhost:8501
                    </button>
                </div>
            </div>
        `;
    }
}

// Function to update loading text
function updateLoadingText(title, description) {
    const loadingTitle = document.querySelector('.loading-title');
    const loadingText = document.querySelector('.loading-text');
    
    if (loadingTitle) loadingTitle.textContent = title;
    if (loadingText) loadingText.textContent = description;
}

// Function to close loading overlay
function closeLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('show');
    }
}

// Function to try redirecting to localhost
function tryRedirect() {
    window.location.href = 'http://localhost:8501';
}

// Add some easter eggs and interactive features
document.addEventListener('keydown', function(event) {
    // Konami code easter egg (up, up, down, down, left, right, left, right, B, A)
    const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];
    if (!window.konamiProgress) window.konamiProgress = 0;
    
    if (event.code === konamiCode[window.konamiProgress]) {
        window.konamiProgress++;
        if (window.konamiProgress === konamiCode.length) {
            // Easter egg activated - add some space magic
            document.body.style.animation = 'rainbow-shift 2s ease-in-out';
            setTimeout(() => {
                document.body.style.animation = '';
            }, 2000);
            window.konamiProgress = 0;
        }
    } else {
        window.konamiProgress = 0;
    }
});

// Add rainbow animation for easter egg
const style = document.createElement('style');
style.textContent = `
    @keyframes rainbow-shift {
        0% { filter: hue-rotate(0deg); }
        25% { filter: hue-rotate(90deg); }
        50% { filter: hue-rotate(180deg); }
        75% { filter: hue-rotate(270deg); }
        100% { filter: hue-rotate(360deg); }
    }
`;
document.head.appendChild(style);
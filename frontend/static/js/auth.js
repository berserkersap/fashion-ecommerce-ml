// Handle login form submission
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const submitButton = e.target.querySelector('button[type="submit"]');
    
    try {
        // Disable button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = 'Signing in...';
        
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.token) {
            // Verify token format
            if (!data.token.startsWith('Bearer ')) {
                throw new Error('Invalid token format');
            }
            
            // Store the token
            localStorage.setItem('authToken', data.token);
            
            // Store user info if provided
            if (data.user) {
                localStorage.setItem('userInfo', JSON.stringify(data.user));
            }
            
            // Redirect to home page or previous page
            const redirectUrl = new URLSearchParams(window.location.search).get('redirect') || '/';
            window.location.href = redirectUrl;
        } else {
            throw new Error(data.error || 'Authentication failed');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert(error.message || 'Login failed. Please try again.');
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = 'Sign in';
    }
});

// Update UI based on auth state
function updateAuthUI() {
    const authToken = localStorage.getItem('authToken');
    const userInfo = JSON.parse(localStorage.getItem('userInfo') || '{}');
    const userSection = document.getElementById('userSection');
    
    if (authToken) {
        userSection.innerHTML = `
            <div class="flex items-center space-x-4">
                <span class="text-gray-700">${userInfo.email || 'User'}</span>
                <button onclick="logout()" 
                        class="text-gray-600 hover:text-gray-900 px-3 py-1 rounded-md border border-gray-300">
                    Logout
                </button>
            </div>
        `;
    } else {
        userSection.innerHTML = `
            <a href="/login" 
               class="text-gray-600 hover:text-gray-900 px-3 py-1 rounded-md border border-gray-300">
                Login
            </a>
        `;
    }
}

// Logout function
async function logout() {
    try {
        // Call backend logout endpoint if exists
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            headers: {
                'Authorization': localStorage.getItem('authToken')
            }
        });
        
        if (!response.ok) {
            console.warn('Logout from backend failed');
        }
    } catch (error) {
        console.warn('Error during logout:', error);
    } finally {
        // Clear local storage
        localStorage.removeItem('authToken');
        localStorage.removeItem('userInfo');
        
        // Redirect to login
        window.location.href = '/login';
    }
}

// Check auth state on page load
document.addEventListener('DOMContentLoaded', () => {
    updateAuthUI();
    
    // Check token expiration
    const token = localStorage.getItem('authToken');
    if (token) {
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            if (payload.exp * 1000 < Date.now()) {
                // Token expired
                logout();
            }
        } catch (error) {
            console.error('Error checking token:', error);
            logout();
        }
    }
}); 
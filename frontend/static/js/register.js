// Registration form handler
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const submitButton = e.target.querySelector('button[type="submit"]');
    
    // Basic form validation
    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }
    
    try {
        // Disable button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = 'Creating Account...';
        
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.token) {
            // Store the token
            localStorage.setItem('authToken', data.token);
            
            // Store user info if provided
            if (data.user) {
                localStorage.setItem('userInfo', JSON.stringify(data.user));
            }
            
            // Redirect to home page
            window.location.href = '/';
        } else {
            throw new Error(data.error || 'Registration failed');
        }
    } catch (error) {
        console.error('Registration error:', error);
        alert(error.message || 'Registration failed. Please try again.');
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = 'Create Account';
    }
});

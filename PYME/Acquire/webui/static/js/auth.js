/**
 * Client-side authentication handling for PYME Acquire
 * Handles user state, login/logout, and cookie management
 */

// Helper function to get cookie value
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

// Helper function to delete cookie
function deleteCookie(name) {
    document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
}

// Check authentication status and update UI
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/user', {
            method: 'GET',
            credentials: 'same-origin'
        });
        
        const data = await response.json();
        
        if (data.authenticated) {
            showAuthenticatedState(data.user);
        } else {
            showUnauthenticatedState();
        }
    } catch (error) {
        console.error('Error checking auth status:', error);
        showUnauthenticatedState();
    }
}

// Show UI for authenticated user
function showAuthenticatedState(userName) {
    const userDisplay = document.getElementById('userDisplay');
    const userNameSpan = document.getElementById('userName');
    const signOutBtn = document.getElementById('signOutBtn');
    const signInBtn = document.getElementById('signInBtn');
    
    if (userDisplay && userNameSpan && signOutBtn && signInBtn) {
        userNameSpan.textContent = userName;
        userDisplay.style.display = 'inline';
        signOutBtn.style.display = 'inline';
        signInBtn.style.display = 'none';
    }
}

// Show UI for unauthenticated user
function showUnauthenticatedState() {
    const userDisplay = document.getElementById('userDisplay');
    const signOutBtn = document.getElementById('signOutBtn');
    const signInBtn = document.getElementById('signInBtn');
    
    if (userDisplay && signOutBtn && signInBtn) {
        userDisplay.style.display = 'none';
        signOutBtn.style.display = 'none';
        signInBtn.style.display = 'inline';
    }
}

// Handle logout
async function handleLogout(e) {
    e.preventDefault();
    
    try {
        const response = await fetch('/logout', {
            method: 'GET',
            credentials: 'same-origin'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Clear the auth cookie
            deleteCookie('auth');
            
            // Redirect to login page
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
        }
    } catch (error) {
        console.error('Error during logout:', error);
        // Still clear cookie and redirect on error
        deleteCookie('auth');
        window.location.href = '/login.html';
    }
}

// Initialize authentication on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have auth elements (main page)
    const signOutBtn = document.getElementById('signOutBtn');
    if (signOutBtn) {
        signOutBtn.addEventListener('click', handleLogout);
        checkAuthStatus();
    }
});

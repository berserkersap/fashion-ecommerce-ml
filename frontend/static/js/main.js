// Global state
let selectedImages = [];
let cartItems = [];
let authToken = localStorage.getItem('authToken');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize search form if it exists
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearch);
    }

    // Initialize cart button
    const cartButton = document.getElementById('cartButton');
    if (cartButton) {
        cartButton.addEventListener('click', toggleCart);
    }

    // Update user section in navbar
    updateUserSection();
    
    // Load cart if user is authenticated
    if (authToken) {
        loadCart();
    }
});

// Update user section in the navigation
function updateUserSection() {
    const userSection = document.getElementById('userSection');
    if (!userSection) return;

    if (authToken) {
        // User is logged in
        userSection.innerHTML = `
            <div class="flex items-center space-x-4">
                <a href="/recommendations" class="text-gray-600 hover:text-gray-900">My Recommendations</a>
                <button id="logoutButton" class="text-gray-600 hover:text-gray-900">Logout</button>
            </div>
        `;
        // Add event listener to logout button
        const logoutButton = document.getElementById('logoutButton');
        if (logoutButton) {
            logoutButton.addEventListener('click', handleLogout);
        }
    } else {
        // User is not logged in
        userSection.innerHTML = `
            <div class="flex items-center space-x-4">
                <a href="/login" class="text-gray-600 hover:text-gray-900">Login</a>
                <a href="/register" class="text-gray-600 hover:text-gray-900">Register</a>
            </div>
        `;
    }
}

// Handle logout
async function handleLogout() {
    try {
        if (authToken) {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': authToken
                }
            });
        }
        // Clear local storage
        localStorage.removeItem('authToken');
        localStorage.removeItem('userInfo');
        
        // Update UI
        authToken = null;
        updateUserSection();
        
        // Redirect to home page
        window.location.href = '/';
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// Image Upload Handler
function handleImageUpload(input) {
    const files = Array.from(input.files);
    const previewContainer = document.getElementById('imagePreview');
    
    files.forEach(file => {
        if (selectedImages.length >= 3) {
            alert('Maximum 3 images allowed');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const container = document.createElement('div');
            container.className = 'image-preview-container';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'image-preview';
            
            const removeBtn = document.createElement('div');
            removeBtn.className = 'remove-image';
            removeBtn.innerHTML = '×';
            removeBtn.onclick = () => {
                container.remove();
                selectedImages = selectedImages.filter(i => i !== file);
            };
            
            container.appendChild(img);
            container.appendChild(removeBtn);
            previewContainer.appendChild(container);
            
            selectedImages.push(file);
        };
        reader.readAsDataURL(file);
    });
}

// Search Handler
async function handleSearch(e) {
    e.preventDefault();
    
    // Check if user is authenticated
    if (!authToken) {
        alert('Please login to search products');
        window.location.href = '/login?redirect=/search';
        return;
    }
    
    const searchQuery = document.getElementById('searchQuery').value;
    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.classList.add('loading');
    
    try {
        const formData = new FormData();
        formData.append('query', searchQuery);
        selectedImages.forEach(file => {
            formData.append('images', file);
        });
        
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Authorization': authToken
            },
            body: formData
        });
        
        const data = await response.json();
        displayResults(data.products);
    } catch (error) {
        console.error('Search failed:', error);
        alert('Search failed. Please try again.');
    } finally {
        resultsContainer.classList.remove('loading');
    }
}

// Display Search Results
function displayResults(products) {
    const container = document.getElementById('searchResults');
    container.innerHTML = '';
    
    products.forEach(product => {
        const card = document.createElement('div');
        card.className = 'product-card bg-white rounded-lg shadow-md overflow-hidden';
        
        card.innerHTML = `
            <img src="${product.image_url}" alt="${product.name}" 
                 class="w-full h-48 object-cover">
            <div class="p-4">
                <h3 class="text-lg font-semibold">${product.name}</h3>
                <p class="text-gray-600">${product.description}</p>
                <div class="mt-4 flex justify-between items-center">
                    <span class="text-xl font-bold">$${product.price.toFixed(2)}</span>
                    <button onclick="addToCart(${JSON.stringify(product).replace(/"/g, '&quot;')})" 
                            class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded">
                        Add to Cart
                    </button>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Cart Functions
async function loadCart() {
    try {
        const response = await fetch('/cart', {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });
        const data = await response.json();
        cartItems = data.items;
        updateCartUI();
    } catch (error) {
        console.error('Failed to load cart:', error);
    }
}

async function addToCart(product) {
    try {
        const response = await fetch('/cart/add', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({
                product_id: product.id,
                quantity: 1
            })
        });
        
        if (response.ok) {
            await loadCart();
            alert('Product added to cart!');
        } else {
            throw new Error('Failed to add to cart');
        }
    } catch (error) {
        console.error('Failed to add to cart:', error);
        alert('Failed to add product to cart. Please try again.');
    }
}

function toggleCart() {
    const modal = document.getElementById('cartModal');
    modal.classList.toggle('hidden');
}

function updateCartUI() {
    const cartCount = document.getElementById('cartCount');
    const cartItems = document.getElementById('cartItems');
    const cartTotal = document.getElementById('cartTotal');
    
    cartCount.textContent = cartItems.length;
    
    let total = 0;
    cartItems.innerHTML = cartItems.map(item => {
        total += item.price * item.quantity;
        return `
            <div class="flex justify-between items-center border-b pb-2">
                <div>
                    <h4 class="font-semibold">${item.product.name}</h4>
                    <p class="text-sm text-gray-600">$${item.price.toFixed(2)} × ${item.quantity}</p>
                </div>
                <button onclick="removeFromCart(${item.id})" 
                        class="text-red-500 hover:text-red-700">Remove</button>
            </div>
        `;
    }).join('');
    
    cartTotal.textContent = `$${total.toFixed(2)}`;
}

async function checkout() {
    try {
        const response = await fetch('/checkout', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });
        
        const data = await response.json();
        if (data.checkout_url) {
            window.location.href = data.checkout_url;
        } else {
            throw new Error('No checkout URL received');
        }
    } catch (error) {
        console.error('Checkout failed:', error);
        alert('Failed to initiate checkout. Please try again.');
    }
}
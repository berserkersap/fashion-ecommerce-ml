// Load recommendations on page load
document.addEventListener('DOMContentLoaded', () => {
    loadRecommendations();
});

async function loadRecommendations() {
    const sections = ['personalizedRecommendations', 'trendingItems', 'recentlyViewed'];
    
    // Show loading state
    sections.forEach(section => {
        const container = document.getElementById(section);
        container.innerHTML = `
            <div class="col-span-full flex justify-center items-center py-8">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
        `;
    });
    
    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            throw new Error('Please login to view recommendations');
        }
        
        const response = await fetch('/api/recommendations', {
            headers: {
                'Authorization': token
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load recommendations');
        }
        
        const data = await response.json();
        
        // Update each section
        if (data.personalized && data.personalized.length) {
            displayProducts(data.personalized, 'personalizedRecommendations');
        } else {
            showEmptyState('personalizedRecommendations', 'No personalized recommendations yet. Keep browsing to get better suggestions!');
        }
        
        if (data.trending && data.trending.length) {
            displayProducts(data.trending, 'trendingItems');
        } else {
            showEmptyState('trendingItems', 'No trending items available right now');
        }
        
        if (data.recently_viewed && data.recently_viewed.length) {
            displayProducts(data.recently_viewed, 'recentlyViewed');
        } else {
            showEmptyState('recentlyViewed', 'No recently viewed items');
        }
    } catch (error) {
        console.error('Failed to load recommendations:', error);
        showError(error.message || 'Failed to load recommendations. Please try again later.');
    }
}

function displayProducts(products, containerId) {
    const container = document.getElementById(containerId);
    if (!products || !products.length) {
        showEmptyState(containerId, 'No items to display');
        return;
    }
    
    container.innerHTML = products.map(product => `
        <div class="product-card bg-white rounded-lg shadow-md overflow-hidden transform transition-transform duration-200 hover:scale-105">
            <div class="relative pb-2/3">
                <img src="${product.image_url}" alt="${product.name}" 
                     class="absolute h-full w-full object-cover"
                     onerror="this.src='/static/images/placeholder.jpg'">
            </div>
            <div class="p-4">
                <h3 class="text-lg font-semibold truncate">${product.name}</h3>
                <p class="text-gray-600 text-sm mb-2 line-clamp-2">${product.description}</p>
                <div class="mt-4 flex justify-between items-center">
                    <span class="text-xl font-bold">$${product.price.toFixed(2)}</span>
                    <button onclick="addToCart(${JSON.stringify(product).replace(/"/g, '&quot;')})" 
                            class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors duration-200">
                        Add to Cart
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

function showEmptyState(containerId, message) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="col-span-full text-center py-8">
            <p class="text-gray-500">${message}</p>
        </div>
    `;
}

function showError(message) {
    const sections = ['personalizedRecommendations', 'trendingItems', 'recentlyViewed'];
    sections.forEach(section => {
        const container = document.getElementById(section);
        container.innerHTML = `
            <div class="col-span-full text-center py-8">
                <div class="text-red-600 bg-red-100 p-4 rounded-lg inline-block">
                    ${message}
                </div>
            </div>
        `;
    });
}

async function addToCart(product) {
    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            window.location.href = '/login?redirect=/recommendations';
            return;
        }
        
        const response = await fetch('/cart/add', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': token
            },
            body: JSON.stringify({
                product_id: product.id,
                quantity: 1
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to add item to cart');
        }
        
        // Update cart count
        const data = await response.json();
        const cartCount = document.getElementById('cartCount');
        if (cartCount) {
            cartCount.textContent = data.cart_count || '0';
        }
        
        // Show success message
        alert('Item added to cart successfully!');
    } catch (error) {
        console.error('Failed to add to cart:', error);
        alert(error.message || 'Failed to add item to cart. Please try again.');
    }
} 
document.getElementById('searchForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent form submission

    // Get the search query from the input field
    var searchQuery = document.getElementById('searchQuery').value;

    // Call a function to perform the search and display results
    performSearch(searchQuery);
});

function performSearch(searchQuery) {
    // Empty if, else if, and else statement
    var resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = ''; // Clear previous results

    if (searchQuery === 'it is good') {
        // Manually assign result for condition 1
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'since good is there';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    } else if (searchQuery === 'it is bad') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'since bad is there';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    } 
    else if (searchQuery === 'it is not good') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'since not good which means bad';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'it is not bad') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'since not bad which means good';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Oh great, another \'life-changing\' gadget. Can\'t wait to add this to my collection of revolutionary paperweights') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'It expresses a sarcastic and dismissive tone towards the described gadget, suggesting that the speaker is not impressed and views it cynically as just another item that won\'t live up to its purported significance.';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'This instant coffee claims to be gourmet. Tastes just like the tears of a disappointed barista exquisite') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = ' It uses sarcasm to express that the instant coffee, despite claiming to be gourmet, is likened to the "tears of a disappointed barista," suggesting that the speaker finds the taste disappointing or unimpressive.';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Bought a \'smart\' toaster. Now my breakfast is as intelligent as my decision to buy a \'smart\' toaster') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = ' The speaker is using sarcasm to suggest that the purchase of a \'smart\' toaster was not a wise decision ';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Oh, great idea! Let\'s all stay here and freeze to death') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'negative';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'The sarcasm in this statement lies in the exaggerated negativity toward the idea of staying in a cold place. However, the underlying positive message is that it\'s not a good idea to stay in the cold';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Sure, why not buy another pair of shoes? Because, clearly, my closet doesn\'t have enough space for the ones I never wear') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'the sarcasm is to playfully suggest that the person has enough shoes and might consider spending their money on something more practical or useful';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Absolutely, add more items to your cart. Because who needs savings or a balanced budget anyway?') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = ' the underlying positive intention is to gently remind the shopper to be mindful of their budget and make thoughtful purchasing decisions. It\'s a playful way of encouraging responsible spending habits';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Bravo! Keep scrolling and filling up that cart. We all know financial stability is overrated') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = ' the positive intention is to humorously encourage the person to reconsider their choices and be mindful of their spending habits.';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else if (searchQuery === 'Oh, go ahead, buy everything you see. Who needs a savings account when you can have a collection of impulse purchases?') {
        // Manually assign result for condition 2
        var resultItem = document.createElement('div');
        resultItem.classList.add('result-item');

        var titleElement = document.createElement('h3');
        titleElement.textContent = 'positive';

        var descriptionElement = document.createElement('p');
        descriptionElement.textContent = 'The positive intention behind the sarcasm is to humorously suggest that it might be beneficial to prioritize thoughtful purchases and savings over impulsive buying, encouraging a more financially responsible approach to shopping.';

        resultItem.appendChild(titleElement);
        resultItem.appendChild(descriptionElement);

        resultsContainer.appendChild(resultItem);
    }
    else {
        // Manually assign a default result if no condition is met
        resultsContainer.innerHTML = 'No results found for the given search query.';
    }
}

function goBack() {
    window.history.back();
}

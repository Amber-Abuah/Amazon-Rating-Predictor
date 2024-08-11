from MultinominalModel import predict_rating
from bs4 import BeautifulSoup
import requests
import gradio as gr

max_review_count = 5

example_urls = [
    "https://www.amazon.co.uk/Trintion-Scratching-Scratcher-Activity-Dangling/dp/B08FT54NRM",
    "https://www.amazon.co.uk/Indoor-Hanging-playing-sleeping-suitable/dp/B0BTVW7G66",
    "https://www.amazon.co.uk/PlayStation-5-Digital-Console-Slim/dp/B0CM9VKQ5N",
    "https://www.amazon.co.uk/Celebrations-Chocolate-Chocolates-Centerpiece-Maltesers/dp/B07L8D6XM8",
    "https://www.amazon.co.uk/HyRich-SIM-Free-Unlocked-Smartphone-Bluetooth-Note-80-Black/dp/B0BG5KBMYK",
    "https://www.amazon.co.uk/Hama-HS-P350-headset-Binaural-Plastic/dp/B07ZR24KQZ",
    "https://www.amazon.co.uk/Skinapeel-Sonic-Facial-Cleanser-Replaceable/dp/B011V6FUG0",
    "https://www.amazon.co.uk/dp/B0BX47X1K9/"
]

def scrape_amazon_reviews(url):
    headers = { "accept-language": "en-GB,en;q=0.9",  
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content)

    # Retrieve image from product page
    image = soup.select_one('#landingImage').attrs.get('src')

    reviews = soup.select("div.review")

    # Extract review description, rating, and predict a rating from the model
    output_reviews = []
    for i in range(min(len(reviews), max_review_count)):
        review_text = reviews[i].select_one("span.review-text").text.replace("The media could not be loaded.", "").strip("Read more").strip("\n")
        rating = reviews[i].select_one("i.review-rating").text.replace("out of 5 stars", "") 
        predicted_rating = predict_rating(review_text)
        output_reviews.append(review_text + "\n\nPredicted Rating: " + str(predicted_rating)[1] + ".0\nActual Rating: " + rating)

    # If there aren't enough reviews, leave the remaining review text boxes empty
    while(len(output_reviews)) < max_review_count:
        output_reviews.append("")

    output_reviews.append(image)
    return output_reviews 

# Main gradio app
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="Amazon URL")
            button = gr.Button(variant="primary")
            gr.Examples(inputs=url, examples=example_urls)
        with gr.Column():
            reviews = [gr.Text(label="Review " + str(i + 1)) for i in range(max_review_count)]
            image = gr.Image(label="Amazon Product Image", interactive=False)

        
    button.click(fn=scrape_amazon_reviews, inputs=url, outputs=reviews + [image])

demo.launch(share=True)
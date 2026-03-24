import pickle
import os
import numpy as np
import pandas as pd
import gradio as gr
from train import predict

# Create Gradio interface
with gr.Blocks(title="Email Spam Classifier", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 📧 Email Spam Classifier
        Classify emails as **Ham (Legitimate)**, **Spam**, **Phishing**, or **Nigerian 419 Scam**
        
        Uses XGBoost with advanced email and URL features, plus text embeddings.
        """
    )
    
    with gr.Row():
        with gr.Column():
            sender_name = gr.Textbox(
                label="Sender Name",
                placeholder="John Doe",
                lines=1
            )
            sender_email = gr.Textbox(
                label="Sender Email",
                placeholder="john.doe@example.com",
                lines=1
            )
            receiver_email = gr.Textbox(
                label="Receiver Email",
                placeholder="recipient@example.com",
                lines=1
            )
            subject = gr.Textbox(
                label="Email Subject",
                placeholder="Meeting Tomorrow",
                lines=1
            )
            body = gr.Textbox(
                label="Email Body",
                placeholder="Enter the full email content here...",
                lines=8
            )
            
            with gr.Row():
                clear_btn = gr.ClearButton(
                    [sender_name, sender_email, receiver_email, subject, body]
                )
                submit_btn = gr.Button("🔍 Classify Email", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="Prediction Result")
        
    # Examples
    gr.Markdown("### 📋 Try These Examples:")
    
    gr.Examples(
        examples=[
    
            # ================= HAM =================
            [
                "Sylvia Hu",
                "sylvia.hu@enron.com",
                "felecia.acevedo@enron.com",
                "FW: Daily Labor Report",
                "Highlights and table of contents for daily labor report including workplace policies and economic updates."
            ],
            [
                "Michael Parker",
                "ivqrnai@pobox.com",
                "xrh@spamassassin.apache.org",
                "Re: svn commit discussion",
                "Discussion about removing .so domain from list and improving spam filtering logic."
            ],
    
            # ================= SPAM =================
            [
                "Daily Deals",
                "offers@discount-now.com",
                "user@gmail.com",
                "🔥 Limited Offer! Buy Now",
                "Exclusive offer just for you! Click here to get huge discounts http://cheap-deals.xyz"
            ],
            [
                "Mok",
                "iplines1983@icable.ph",
                "user@gvc.ceas.cc",
                "Upgrade your pleasure",
                "Upgrade your sex and pleasures with these techniques http://www.brightmade.com"
            ],
    
            # ================= PHISHING =================
            [
                "Bank Support",
                "support@secure-bank-alert.com",
                "user@gmail.com",
                "Account Suspended - Immediate Action Required",
                "Your account has been suspended. Verify now at http://secure-login-bank.com to avoid permanent closure."
            ],
            [
                "PayPal Security",
                "alert@paypal-secure.com",
                "user@yahoo.com",
                "Unauthorized Login Attempt",
                "We detected unusual activity. Confirm your identity immediately at http://paypal-verification.com"
            ],
    
            # ================= NIGERIAN FRAUD =================
            [
                "Prince Adewale",
                "ade.prince@lagosmail.ng",
                "user@gmail.com",
                "Urgent Business Proposal",
                "I am a Nigerian prince with $10 million. I need your assistance to transfer funds. You will receive 20%."
            ],
            [
                "Dr. Musa Ibrahim",
                "musa.ibrahim@africafinance.ng",
                "user@yahoo.com",
                "Confidential Transfer Opportunity",
                "Funds worth $5.5M are trapped in a bank. Kindly assist and earn a share. This is risk free."
            ]
    
        ],
        inputs=[sender_name, sender_email, receiver_email, subject, body],
        label=None
    )
    
    # Connect the function
    submit_btn.click(
        fn=predict,
        inputs=[sender_name, sender_email, receiver_email, subject, body],
        outputs=output
    )
    
    # Add footer with info
    gr.Markdown(
        """
        ---
        **Note:** Make sure you have trained the model first by running `python train.py` before using this interface.
        
        This classifier uses:
        - 🔍 Email header analysis (sender/receiver patterns)
        - 🔗 URL feature extraction (suspicious patterns, shorteners, IP addresses)
        - 📝 Text embeddings (semantic content analysis)
        - 🤖 XGBoost gradient boosting (optimized hyperparameters)
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)

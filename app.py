# ... [Your Model and Encoding functions stay the same] ...

if st.button("Identify Genes"):
    # 1. AI Analysis
    input_tensor = torch.tensor(one_hot_encode(user_dna)).float().T.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # Define these variables INSIDE the button block
        confidence = probs[0][1].item() * 100
        prediction = torch.argmax(output, dim=1).item()

    # 2. Results Display
    st.write(f"AI Confidence: {confidence:.2f}%")
    
    # 3. Biological Identification (BLAST)
    # This must be indented so it only runs AFTER prediction is defined
    if prediction == 1 and confidence > 90:
        st.success("High-confidence gene detected!")
        gene_info = identify_gene(user_dna) # Using the function provided earlier
        st.subheader("🧬 Biological Identity")
        st.write(gene_info)
    else:
        st.warning("No significant gene detected or confidence too low for identification.")

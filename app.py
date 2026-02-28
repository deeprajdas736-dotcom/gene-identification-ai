from Bio.Blast import NCBIWWW, NCBIXML

def identify_gene(sequence):
    """Sends sequence to NCBI to find its name and function."""
    st.info("Searching NCBI Database for identification... (This may take a minute)")
    
    # Run BLAST online against the 'nt' (nucleotide) database
    result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
    blast_record = NCBIXML.read(result_handle)
    
    if blast_record.alignments:
        # Get the top hit
        top_hit = blast_record.alignments[0]
        gene_name = top_hit.title
        # Attempt to get functional info from the description
        return gene_name
    else:
        return "No known match found in global databases."

# --- Inside your Streamlit 'Analyze' button logic ---
if prediction == 1 and confidence > 90:
    name_and_func = identify_gene(user_dna)
    st.subheader("🧬 Biological Identity")
    st.write(name_and_func)

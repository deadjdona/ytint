"""Stage 15: Named Entity Extraction (s15_ner.py)

Uses spaCy (ru_core_news_sm) to extract people, places, and organizations 
from the comment text, saving them into an entity relational table.
"""

import pandas as pd
from pathlib import Path
from engine.config_loader import load_config
import spacy

def run_ner():
    print("👤 Starting Named Entity Extraction (s15)...")
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    comments_file = interim_dir / "comments_clean.parquet"
    if not comments_file.exists():
        print("⚠️ Missing comments_clean.parquet. Skipping s15.")
        return
        
    print("  -> Loading spaCy model (ru_core_news_sm / xx_ent_wiki_sm / en_core_web_sm)...")
    nlp = None
    for model_name in ["ru_core_news_sm", "xx_ent_wiki_sm", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_name, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            print(f"  -> Successfully loaded spaCy model '{model_name}'.")
            break
        except OSError:
            continue
            
    if nlp is None:
        print("⚠️ No spaCy models found. Falling back to heuristic proper noun entity extractor.")
        
    print("  -> Loading comments...")
    text_col = 'text'
    comments_df = pd.read_parquet(comments_file)
    if 'text' not in comments_df.columns and 'text_original' in comments_df.columns:
        text_col = 'text_original'
    if 'comment_id' not in comments_df.columns:
        comments_df['comment_id'] = [f"c{i}" for i in range(len(comments_df))]
    df = comments_df[['comment_id', text_col]].dropna(subset=[text_col])
    
    if df.empty:
        print("⚠️ No valid text found. Skipping.")
        return
        
    print(f"  -> Extracting entities from {len(df)} comments (this may take a minute)...")
    
    entities_data = []
    
    if nlp is not None:
        for doc, comment_id in zip(nlp.pipe(df[text_col], batch_size=1000), df['comment_id']):
            for ent in doc.ents:
                if ent.label_ in ['PER', 'LOC', 'ORG', 'PERSON']:
                    label = 'PER' if ent.label_ == 'PERSON' else ent.label_
                    entities_data.append({
                        'comment_id': comment_id,
                        'entity_text': ent.text,
                        'entity_label': label
                    })
    else:
        import re
        for row in df.itertuples():
            text_str = str(getattr(row, text_col))
            words = text_str.split()
            if len(words) > 1:
                for w in words[1:]:
                    clean_w = re.sub(r'[^\w]', '', w)
                    if clean_w and clean_w[0].isupper() and not clean_w.isupper() and len(clean_w) > 2:
                        entities_data.append({
                            'comment_id': row.comment_id,
                            'entity_text': clean_w,
                            'entity_label': 'PROPER_NOUN'
                        })
                
    df_entities = pd.DataFrame(entities_data)
    
    out_file = out_dir / "named_entities.parquet"
    if not df_entities.empty:
        df_entities.to_parquet(out_file, index=False)
        print(f"✅ Extracted {len(df_entities)} entities (PER/LOC/ORG).")
    else:
        # Save empty dataframe with correct columns
        pd.DataFrame(columns=['comment_id', 'entity_text', 'entity_label']).to_parquet(out_file, index=False)
        print("✅ Finished extraction, but found 0 entities.")
        
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    run_ner()

import gradio as gr
import tensorflow as tf
import keras
import pickle as pkl




categories = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions',
       'black', 'white', 'identity_any', 'severe_toxicity', 'obscene',
       'threat', 'insult', 'identity_attack', 'sexual_explicit', 'y',
       'from_source_domain']

model = keras.models.load_model("toxicity.keras")
#print("Model loaded")

from_disk = pkl.load(open("tv_layer.pkl", "rb"))
vectorizer = keras.layers.TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk['weights'])
#print("Vectorizer loaded")

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(categories):
         text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


temp = "hey you fuck you mf"
#print(score_comment(temp))
demo = gr.Interface(fn=score_comment, inputs="text", outputs="text")
demo.launch(share=True)   

#print("ok")
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from openai import OpenAI
import re
import os
import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


WEIGHTS_FILE = "node_weights.json"


def build_graph():
    nodes = []
    edges = []

    with open("exported_jsonl/public.node.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            nodes.append(json.loads(line))
            if "embedding" in nodes[i].get("attrs", {}):
                del nodes[i]["attrs"]["embedding"]

    with open("exported_jsonl/public.edge.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))

    graph = nx.Graph()

    for node in nodes:
        graph.add_node(
            node["id"],
            raw=node["attrs"],
            subnet=node.get("subnet"),
            label=node["id"]
        )

    for edge in edges:
        graph.add_edge(edge["a"], edge["b"])

    return graph


def save_weights(weights, weights_file=WEIGHTS_FILE):
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)


def renormalize_weights(weights, target_total):
    current_total = sum(weights.values())

    if current_total <= 0:
        raise ValueError("All node weights are zero or negative.")

    scale = target_total / current_total

    return {
        node: weight * scale
        for node, weight in weights.items()
    }


def create_initial_weights(G, weights_file=WEIGHTS_FILE):
    """
    Initial weight = number of neighbors.

    Isolated nodes get weight 1 instead of 0.
    Then weights are renormalized so the total equals number of nodes.
    """
    weights = {
        str(node): float(max(1, np.sqrt(G.degree[node])))
        for node in G.nodes
    }

    weights = renormalize_weights(weights, target_total=len(G.nodes))

    save_weights(weights, weights_file)

    return weights


def load_or_create_weights(G, weights_file=WEIGHTS_FILE):
    graph_nodes = {str(node) for node in G.nodes}

    if not os.path.exists(weights_file):
        return create_initial_weights(G, weights_file)

    with open(weights_file, "r", encoding="utf-8") as f:
        weights = json.load(f)

    # Add missing nodes using degree-based initial weight
    for node in graph_nodes:
        if node not in weights:
            weights[node] = float(max(1, np.sqrt(G.degree[node])))

    # Remove weights for nodes no longer in graph
    weights = {
        node: float(weight)
        for node, weight in weights.items()
        if node in graph_nodes
    }

    weights = renormalize_weights(weights, target_total=len(G.nodes))

    save_weights(weights, weights_file)

    return weights


def weighted_choice(items, weights, rng):
    return rng.choices(
        population=items,
        weights=weights,
        k=1
    )[0]


def execute_random_traversal(
    G,
    k,
    node_weights,
    decay=0.5,
    weights_file=WEIGHTS_FILE,
    seed=None
):
    rng = random.Random(seed)

    target_total = len(G.nodes)

    start_candidates = list(G.nodes)
    start_weights = [
        node_weights[str(node)]
        for node in start_candidates
    ]

    start_node = weighted_choice(start_candidates, start_weights, rng)

    path = [start_node]
    current = start_node

    for _ in range(k):
        neighbors = list(G.neighbors(current))

        if not neighbors:
            break

        neighbor_weights = [
            node_weights[str(neighbor)]
            for neighbor in neighbors
        ]

        current = weighted_choice(neighbors, neighbor_weights, rng)
        path.append(current)

    visited = {str(node) for node in path}

    for node in visited:
        node_weights[node] *= decay

    node_weights = renormalize_weights(node_weights, target_total)

    save_weights(node_weights, weights_file)

    return path, node_weights


def _get_node_content(G, node_id):
    """Return the relevant content for a node based on its subnet."""
    node = G.nodes[node_id]
    attrs = node.get("raw", {})
    subnet = node.get("subnet")

    if subnet == "Topics":
        return attrs.get("topics") or str(node_id)
    elif subnet == "People":
        return attrs.get("people") or str(node_id)
    elif subnet == "Notes":
        return attrs.get("description") or str(node_id)
    elif subnet == "Problems":
        problem = attrs.get("problem", "")
        solution = attrs.get("solution", "")
        return f"Problem: {problem}\nSolution: {solution}".strip() or str(node_id)
    return str(node_id)


def generate_path_interpretations(G, path_nodes, model="gpt-5.4"):
    """
    For each consecutive pair along path_nodes, ask an LLM to interpret
    how the content of node[i-1] is relevant to the content of node[i].

    Returns a list of (len(path_nodes) - 1) strings, one per step.
    """
    
    client = OpenAI()
    interpretations = []

    for i in range(1, len(path_nodes)):
        prev_content = _get_node_content(G, path_nodes[i - 1])
        curr_content = _get_node_content(G, path_nodes[i])

        prompt = (
            f"You are an expert at connecting ideas across different domains.\n\n"
            f"Note A:\n{prev_content}\n\n"
            f"Note B:\n{curr_content}\n\n"
            f"In a few paragraphs, explain how Note A is relevant to Note B — "
            f"what conceptual bridge links them?"
        )

        response = client.chat.completions.create(
            model=model,
            reasoning_effort="high",
            messages=[{"role": "user", "content": prompt}],
        )
        interpretations.append(response.choices[0].message.content.strip())

    return interpretations


def generate_podcast_script(interpretations, model="gpt-5.4"):
    """
    Given the ordered list of interpretation strings produced by
    generate_path_interpretations, generate a podcast-style dialogue
    between two hosts (Alex and Jordan) that covers each interpretation
    in order.

    Returns the full script as a single string.
    """
    client = OpenAI()

    numbered = "\n\n".join(
        f"[Segment {i + 1}]\n{text}"
        for i, text in enumerate(interpretations)
    )

    system_prompt = (
        """ 
        You are an expert scriptwriter creating a highly engaging, intellectual podcast, similar to an AI "Audio Overview." 
        The two hosts are Alex (male) and Jordan (female). They are exceptionally curious, warm, and enthusiastic about learning. 
        Their dynamic is highly conversational and realistic: they use natural filler words ("yeah," "wow," "exactly," "wait, so..."), supportively interrupt or build on each other's sentences, and frequently use relatable analogies to explain complex ideas. 
        Crucially, they treat the listener like a smart friend who is revisiting a fascinating topic after a long time. They don't assume zero knowledge, but they DO assume the listener needs a thorough, engaging refresher. They are masters at dusting off familiar but fuzzy concepts, re-laying the groundwork, and bringing the listener fully back up to speed before diving deep.
        Format the output strictly with their names followed by a colon (e.g., 'Alex:' and 'Jordan:'). Do not include stage directions, parentheticals, or sound effects—rely solely on the dialogue to convey tone and energy.
        """
    )

    user_prompt = (
        f"""
        Below is the source material for today's episode, broken down into specific segments in strict chronological order. 
        Write a podcast script where Alex and Jordan discuss EVERY single segment in the EXACT order provided. 

        CRITICAL CONSTRAINTS:
        1. The Thorough Refresher: Introduce every concept with the vibe of "you probably remember this, but let's quickly refresh." Re-define essential terms and thoroughly re-establish the basic context so the listener gets a complete recap before getting into the weeds.
        2. The "Summarize, Elaborate, Connect" Loop: For EVERY segment, the conversation must follow this pacing:
        - SUMMARIZE: First, re-introduce and state the core concept of the current segment clearly to jog the memory.
        - ELABORATE: Next, spend significant time unpacking the details. Have one host act as the curious sounding board asking clarifying questions, while the other breaks it down using relatable, everyday analogies to make the old concepts feel fresh. Deeply explore the idea before moving on.
        - CONNECT: Only AFTER the segment has been fully refreshed and elaborated on, create a natural, conversational bridge (e.g., "And if you remember that, then this next piece suddenly clicks...", "Which actually brings us back to...") to transition into the subsequent segment.
        3. Strict Sequential Adherence: Do not skip, reorder, or merge any segments. You must hit every single point individually.
        4. Seamless Flow: Do not explicitly mention the segment numbers in the dialogue. Disguise the structural loop within a continuous, engrossing, and natural conversation.

        Here is the material they are revisiting today:

        {numbered}
        """
    )

    response = client.chat.completions.create(
        model=model,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

def generate_podcast(script_text, output_file="podcast.mp3"):
    """
    Parses a two-host podcast script and generates a single audio file using OpenAI TTS.
    """
    # Initialize the OpenAI client
    client = OpenAI()

    # Map the hosts to specific OpenAI voices
    voice_mapping = {
        "Alex": "onyx",
        "Jordan": "nova"
    }

    # Regex pattern to capture the speaker's name and their dialogue.
    # It looks for "Name:", then captures everything until the next "Name:" or the end of the text.
    pattern = r"(Alex|Jordan):\s*(.*?)(?=(?:Alex|Jordan):|$)"
    
    # re.DOTALL ensures it captures multi-line dialogue blocks
    segments = re.findall(pattern, script_text, re.DOTALL | re.IGNORECASE)

    if not segments:
        print("No dialogue segments found. Please check the script format.")
        return

    print(f"Found {len(segments)} segments. Beginning audio generation...")

    # Open the final output file in binary append mode
    with open(output_file, "wb") as f_out:
        for index, (speaker, text) in enumerate(segments):
            # Clean up the parsed strings
            speaker = speaker.capitalize().strip()
            text = text.strip()
            
            if not text:
                continue

            # Fallback to 'alloy' if the speaker name isn't in our mapping
            voice = voice_mapping.get(speaker, "alloy")
            
            print(f"[{index + 1}/{len(segments)}] Generating audio for {speaker}...")

            try:
                # Call the OpenAI API for this specific block of text
                # Using 'tts-1' for faster, cost-effective generation. Use 'tts-1-hd' for higher quality.
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text
                )
                
                # Stream the binary audio data directly into our final MP3 file
                for chunk in response.iter_bytes():
                    f_out.write(chunk)
                    
            except Exception as e:
                print(f"Error generating audio for segment {index + 1}: {e}")
                break

    print(f"\nSuccess! Podcast saved to {output_file}")


def run_podcast_pipeline(
    k=10,
    decay=0.5,
    seed=None,
    weights_file=WEIGHTS_FILE,
):
    """
    End-to-end pipeline: build graph, traverse it, generate interpretations,
    write a podcast script, and render it to an MP3.
    """
    G = build_graph()
    node_weights = load_or_create_weights(G, weights_file)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = f'KB_Traversal_{today_str}.mp3'
    folder_id = '1S5EI7dgTbLsIeLwUFE1W9uZGLdITYH4P'

    path_nodes, _ = execute_random_traversal(
        G,
        k=k,
        node_weights=node_weights,
        decay=decay,
        weights_file=weights_file,
        seed=seed,
    )

    interpretations = generate_path_interpretations(G, path_nodes)
    script = generate_podcast_script(interpretations)
    generate_podcast(script, output_file)


    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    
    # 1. Check if we already have a saved session token
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # 2. If there are no valid credentials, log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next automated run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    # 3. Build the service and upload
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': output_file,
        'parents': [folder_id],
    }

    media = MediaFileUpload(output_file, mimetype='audio/mpeg', resumable=True)
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f"Success! Uploaded to Drive with File ID: {uploaded.get('id')}")
    os.remove(output_file)
    print(f"Local file '{output_file}' deleted.")


if __name__ == "__main__":
    run_podcast_pipeline()

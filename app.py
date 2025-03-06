import os
import sys
sys.path.append(".")
sys.path.append("./tools")
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json
import pandas as pd
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, runtime_logging
from typing import List, Dict, Tuple

from tools.manager_map import MapController, calculate_map_bounds
from tools.manager_db import DBManager
from tools.manager_online import OnlineManager

from agents import create_agents
from streamlit_image_select import image_select  # Import the image select widget


def llm_tab():
    dbcontroller = DBManager()
    onlinecontroller = OnlineManager()
    if 'controller' not in st.session_state:
        st.session_state.controller = MapController()
        st.session_state.messages = []
        st.session_state.function_calls = []
        st.session_state.detailed_logs = []
        st.session_state.assistant, st.session_state.user_proxy = create_agents(
            st.session_state.controller, dbcontroller, onlinecontroller
        )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Column 1: Chat control
    with col1:
        messages_container = st.container(height=600)
        with messages_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        prompt = st.chat_input("Ask a question or control the map...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            try:
                # Store detailed logs for this interaction
                current_logs = []
                import builtins
                original_print = builtins.print
                
                def custom_print(*args, **kwargs):
                    original_print(*args, **kwargs)
                    output = " ".join(str(arg) for arg in args)
                    current_logs.append(output)
                
                builtins.print = custom_print

                # Format the conversation history (excluding the current prompt)
                conversation_history = format_messages_for_llm(st.session_state.messages[:-1])
                full_prompt = f"{conversation_history} Human: {prompt}"

                # Get agent response
                response = st.session_state.user_proxy.initiate_chat(
                    st.session_state.assistant,
                    message=full_prompt
                )

                # Restore the original print function
                builtins.print = original_print
                
                # Store the logs
                st.session_state.detailed_logs.append({
                    "prompt": prompt,
                    "logs": current_logs
                })
                
                chat_history = st.session_state.user_proxy.chat_messages[st.session_state.assistant]
                for message in chat_history:
                    if message["role"] == "user" and message["content"] and message["content"] != prompt:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
                        st.session_state.messages.append({"role": "assistant", "content": message["content"]})
                        break
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    # Column 2: Map display
    with col2:
        m = folium.Map(location=st.session_state.controller.map_center, 
                      zoom_start=st.session_state.controller.zoom)
        
        if st.session_state.controller.map_type == "Terrain":
            folium.TileLayer('Stamen Terrain', opacity=st.session_state.controller.layer_opacity).add_to(m)
        elif st.session_state.controller.map_type == "Satellite":
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                opacity=st.session_state.controller.layer_opacity
            ).add_to(m)
        
        # Add regular markers
        for marker in st.session_state.controller.markers:
            folium.Marker(
                marker["pos"],
                popup=marker["label"] if marker["label"] else None
            ).add_to(m)
        
        # Add cluster markers
        for cluster_marker in st.session_state.controller.cluster_markers:
            folium.Marker(
                cluster_marker["pos"],
                popup="geo flood cluster"
            ).add_to(m)
        
        # Add flood data markers if enabled
        if st.session_state.controller.show_flood_data:
            filtered_df = st.session_state.controller.flood_data[
                (st.session_state.controller.flood_data['flood_confidence'].between(
                    st.session_state.controller.intensity_range[0], 
                    st.session_state.controller.intensity_range[1]
                )) &
                (st.session_state.controller.flood_data['feature_type'].isin(
                    st.session_state.controller.selected_features
                ))
            ]
            for _, row in filtered_df.iterrows():
                popup_content = f"""
                    Feature Type: {row['feature_type']}<br>
                    Flood Intensity: {row['flood_confidence']:.3f}
                """
                def get_color(intensity):
                    if intensity < 0.3:
                        return 'green'
                    elif intensity < 0.5:
                        return 'yellow'
                    elif intensity < 0.7:
                        return 'orange'
                    else:
                        return 'red'
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=popup_content,
                    color=get_color(row['flood_confidence']),
                    fill=True,
                    fillColor=get_color(row['flood_confidence']),
                    fillOpacity=0.7
                ).add_to(m)
            
            legend_html = """
            <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                        padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <p><strong>Flood Intensity</strong></p>
                <p><i style="background: red; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> &gt; 0.7</p>
                <p><i style="background: orange; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> 0.5 - 0.7</p>
                <p><i style="background: yellow; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> 0.3 - 0.5</p>
                <p><i style="background: green; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> &lt; 0.3</p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        Draw(export=True).add_to(m)

        if 'previous_draw_info' not in st.session_state:
            st.session_state.previous_draw_info = None
        current_draw_info = st_folium(m, width=800, height=600, key="llm_map")
        if current_draw_info != st.session_state.previous_draw_info:
            st.session_state.controller.draw_info = current_draw_info
            st.session_state.previous_draw_info = current_draw_info
            
        # Dropdown for markers
        if st.session_state.controller.markers:
            st.write("### Current Markers")
            marker_labels = [
                f"{m['label']} ({m['pos'][0]:.4f}, {m['pos'][1]:.4f})" if m['label'] 
                else f"({m['pos'][0]:.4f}, {m['pos'][1]:.4f})" 
                for m in st.session_state.controller.markers
            ]
            selected_marker = st.selectbox("Select a marker to view details:", marker_labels, key="llm_marker_select")
            if selected_marker:
                idx = marker_labels.index(selected_marker)
                marker = st.session_state.controller.markers[idx]
                st.write(f"**Label:** {marker['label'] if marker['label'] else 'No label'}")
                st.write(f"**Latitude:** {marker['pos'][0]:.6f}")
                st.write(f"**Longitude:** {marker['pos'][1]:.6f}")

    # Column 3: Image selection and Interaction log
    with col3:
        st.title("Select an Image 🖼️")
        # Define a list of image URLs (or file paths)
        image_files = [
            "https://placekitten.com/200/300",
            "https://placekitten.com/250/300",
            "https://placekitten.com/300/300"
        ]
        # The image_select widget returns the selected image URL/path.
        selected_image = image_select("Choose an image:", image_files, use_container_width=True)
        if selected_image:
            st.image(selected_image, caption="You selected this image!", use_column_width=True)
            st.session_state.selected_image = selected_image
            st.write(f"You selected: {selected_image}")
        else:
            st.write("No image selected yet.")
        
        st.markdown("---")
        st.title("Interaction Log 📝")
        if st.session_state.detailed_logs:
            recent_logs = st.session_state.detailed_logs[-3:]
            for idx, interaction in enumerate(reversed(recent_logs), 1):
                with st.expander(f"Recent Interaction {idx}", expanded=True):
                    st.markdown("💭 **User Query:**")
                    st.write(interaction['prompt'])
                    
                    st.markdown("🔄 **Actions:**")
                    for log in interaction['logs']:
                        clean_log = log.replace('[32m', '').replace('[0m', '')
                        if ">>>>>>>> EXECUTING FUNCTION" in clean_log:
                            function_name = clean_log.split('EXECUTING FUNCTION')[1].split('...')[0].strip()
                            st.write(f"⚡ EXECUTING FUNCTION {function_name}...")
                        elif "Response from calling function" in clean_log:
                            response = clean_log.split("*****")[1] if len(clean_log.split("*****")) > 1 else clean_log
                            st.write("✅ **User Proxy**:")
                            st.write(response.strip())
                        elif "map_assistant (to user_proxy):" in clean_log and \
                             "Suggested function call" not in clean_log and \
                             "Response from calling" not in clean_log:
                            response = clean_log.replace("map_assistant (to user_proxy):", "").strip()
                            if not response.startswith("*****"):
                                st.markdown("💬 **Assistant's Response:**")
                                st.write(response)


def format_messages_for_llm(messages: List[Dict[str, str]]) -> str:
    """Format message history into a single string for the LLM."""
    formatted = ""
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n\n"
    return formatted

def main():
    st.set_page_config(layout="wide")
    tab1, tab2 = st.tabs(["LLM Control", "Manual Map Control"])
    with tab1:
        llm_tab()

if __name__ == "__main__":
    main()

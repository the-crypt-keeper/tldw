# Local_LLM_Inference_Engine_Lib.py
#########################################
# Local LLM Inference Engine Library
# This library is used to handle downloading, configuring, and launching the Local LLM Inference Engine
#   via (llama.cpp via llamafile)
#
#
####
####################
# Function List
#
# 1. download_latest_llamafile(repo, asset_name_prefix, output_filename)
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. cleanup_process()
# 5. signal_handler(sig, frame)
# 6. local_llm_function()
# 7. launch_in_new_terminal_windows(executable, args)
# 8. launch_in_new_terminal_linux(executable, args)
# 9. launch_in_new_terminal_mac(executable, args)
#
####################
# Import necessary libraries
from asyncio import subprocess
import atexit
import re
import sys
import time
# Import 3rd-pary Libraries
#
# Import Local
from Article_Summarization_Lib import *
from App_Function_Libraries.Utils import download_file
#
#
#######################################################################################################################
# Function Definitions
#

# Download latest llamafile from Github
    # Example usage
    #repo = "Mozilla-Ocho/llamafile"
    #asset_name_prefix = "llamafile-"
    #output_filename = "llamafile"
    #download_latest_llamafile(repo, asset_name_prefix, output_filename)

# THIS SHOULD ONLY BE CALLED IF THE USER IS USING THE GUI TO SETUP LLAMAFILE
# Function is used to download only llamafile
def download_latest_llamafile_no_model(output_filename):
    # Check if the file already exists
    print("Checking for and downloading Llamafile it it doesn't already exist...")
    if os.path.exists(output_filename):
        print("Llamafile already exists. Skipping download.")
        logging.debug(f"{output_filename} already exists. Skipping download.")
        llamafile_exists = True
    else:
        llamafile_exists = False

    if llamafile_exists == True:
        pass
    else:
        # Establish variables for Llamafile download
        repo = "Mozilla-Ocho/llamafile"
        asset_name_prefix = "llamafile-"
        # Get the latest release information
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(latest_release_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch latest release info: {response.status_code}")

        latest_release_data = response.json()
        tag_name = latest_release_data['tag_name']

        # Get the release details using the tag name
        release_details_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
        response = requests.get(release_details_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch release details for tag {tag_name}: {response.status_code}")

        release_data = response.json()
        assets = release_data.get('assets', [])

        # Find the asset with the specified prefix
        asset_url = None
        for asset in assets:
            if re.match(f"{asset_name_prefix}.*", asset['name']):
                asset_url = asset['browser_download_url']
                break

        if not asset_url:
            raise Exception(f"No asset found with prefix {asset_name_prefix}")

        # Download the asset
        response = requests.get(asset_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download asset: {response.status_code}")

        print("Llamafile downloaded successfully.")
        logging.debug("Main: Llamafile downloaded successfully.")

        # Save the file
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        logging.debug(f"Downloaded {output_filename} from {asset_url}")
        print(f"Downloaded {output_filename} from {asset_url}")
    return output_filename


# FIXME - Add option in GUI for selecting the other models for download
# Should only be called from 'local_llm_gui_function' - if its called from anywhere else, shits broken.
# Function is used to download llamafile + A model from Huggingface
def download_latest_llamafile_through_gui(repo, asset_name_prefix, output_filename):
    # Check if the file already exists
    print("Checking for and downloading Llamafile it it doesn't already exist...")
    if os.path.exists(output_filename):
        print("Llamafile already exists. Skipping download.")
        logging.debug(f"{output_filename} already exists. Skipping download.")
        llamafile_exists = True
    else:
        llamafile_exists = False

    if llamafile_exists == True:
        pass
    else:
        # Get the latest release information
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(latest_release_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch latest release info: {response.status_code}")

        latest_release_data = response.json()
        tag_name = latest_release_data['tag_name']

        # Get the release details using the tag name
        release_details_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
        response = requests.get(release_details_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch release details for tag {tag_name}: {response.status_code}")

        release_data = response.json()
        assets = release_data.get('assets', [])

        # Find the asset with the specified prefix
        asset_url = None
        for asset in assets:
            if re.match(f"{asset_name_prefix}.*", asset['name']):
                asset_url = asset['browser_download_url']
                break

        if not asset_url:
            raise Exception(f"No asset found with prefix {asset_name_prefix}")

        # Download the asset
        response = requests.get(asset_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download asset: {response.status_code}")

        print("Llamafile downloaded successfully.")
        logging.debug("Main: Llamafile downloaded successfully.")

        # Save the file
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        logging.debug(f"Downloaded {output_filename} from {asset_url}")
        print(f"Downloaded {output_filename} from {asset_url}")

    # Check to see if the LLM already exists, and if not, download the LLM
    print("Checking for and downloading LLM from Huggingface if needed...")
    logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
    mistral_7b_instruct_v0_2_q8_0_llamafile = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
    Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8 = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
    Phi_3_mini_128k_instruct_Q8_0_gguf = "Phi-3-mini-128k-instruct-Q8_0.gguf"
    if os.path.exists(mistral_7b_instruct_v0_2_q8_0_llamafile):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    elif os.path.exists(Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    elif os.path.exists(mistral_7b_instruct_v0_2_q8_0_llamafile):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    else:
        logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
        print("Downloading LLM from Huggingface...")
        time.sleep(1)
        print("Gonna be a bit...")
        time.sleep(1)
        print("Like seriously, an 8GB file...")
        time.sleep(2)
        # Not needed for GUI
        # dl_check = input("Final chance to back out, hit 'N'/'n' to cancel, or 'Y'/'y' to continue: ")
        #if dl_check == "N" or dl_check == "n":
        #     exit()
        x = 2
        if x != 1:
            print("Uhhhh how'd you get here...?")
            exit()
        else:
            print("Downloading LLM from Huggingface...")
            # Establish hash values for LLM models
            mistral_7b_instruct_v0_2_q8_gguf_sha256 = "f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06"
            samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
            mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
            global llm_choice

            # FIXME - llm_choice
            llm_choice = 2
            llm_choice = input("Which LLM model would you like to download? 1. Mistral-7B-Instruct-v0.2-GGUF or 2. Samantha-Mistral-Instruct-7B-Bulleted-Notes) (plain or 'custom') or MS Flavor: Phi-3-mini-128k-instruct-Q8_0.gguf  \n\n\tPress '1' or '2' or '3' to specify: ")
            while llm_choice != "1" and llm_choice != "2" and llm_choice != "3":
                print("Invalid choice. Please try again.")
            if llm_choice == "1":
                llm_download_model = "Mistral-7B-Instruct-v0.2-Q8.llamafile"
                mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
                llm_download_model_hash = mistral_7b_instruct_v0_2_q8_0_llamafile_sha256
                llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
                llamafile_llm_output_filename = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "2":
                llm_download_model = "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf"
                samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
                llm_download_model_hash = samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256
                llamafile_llm_output_filename = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes-GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "3":
                llm_download_model = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                Phi_3_mini_128k_instruct_Q8_0_gguf_sha256 = "6817b66d1c3c59ab06822e9732f0e594eea44e64cae2110906eac9d17f75d193"
                llm_download_model_hash = Phi_3_mini_128k_instruct_Q8_0_gguf_sha256
                llamafile_llm_output_filename = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/gaianet/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct-Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "4": # FIXME - and meta_Llama_3_8B_Instruct_Q8_0_llamafile_exists == False:
                meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256 = "406868a97f02f57183716c7e4441d427f223fdbc7fa42964ef10c4d60dd8ed37"
                llm_download_model_hash = meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256
                llamafile_llm_output_filename = " Meta-Llama-3-8B-Instruct.Q8_0.llamafile"
                llamafile_llm_url = "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.llamafile?download=true"
            else:
                print("Invalid choice. Please try again.")
    return output_filename


# Maybe replace/ dead code? FIXME
# Function is used to download llamafile + A model from Huggingface
def download_latest_llamafile(repo, asset_name_prefix, output_filename):
    # Check if the file already exists
    print("Checking for and downloading Llamafile it it doesn't already exist...")
    if os.path.exists(output_filename):
        print("Llamafile already exists. Skipping download.")
        logging.debug(f"{output_filename} already exists. Skipping download.")
        llamafile_exists = True
    else:
        llamafile_exists = False

    if llamafile_exists == True:
        pass
    else:
        # Get the latest release information
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(latest_release_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch latest release info: {response.status_code}")

        latest_release_data = response.json()
        tag_name = latest_release_data['tag_name']

        # Get the release details using the tag name
        release_details_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
        response = requests.get(release_details_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch release details for tag {tag_name}: {response.status_code}")

        release_data = response.json()
        assets = release_data.get('assets', [])

        # Find the asset with the specified prefix
        asset_url = None
        for asset in assets:
            if re.match(f"{asset_name_prefix}.*", asset['name']):
                asset_url = asset['browser_download_url']
                break

        if not asset_url:
            raise Exception(f"No asset found with prefix {asset_name_prefix}")

        # Download the asset
        response = requests.get(asset_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download asset: {response.status_code}")

        print("Llamafile downloaded successfully.")
        logging.debug("Main: Llamafile downloaded successfully.")

        # Save the file
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        logging.debug(f"Downloaded {output_filename} from {asset_url}")
        print(f"Downloaded {output_filename} from {asset_url}")

    # Check to see if the LLM already exists, and if not, download the LLM
    print("Checking for and downloading LLM from Huggingface if needed...")
    logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
    mistral_7b_instruct_v0_2_q8_0_llamafile = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
    Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8 = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
    Phi_3_mini_128k_instruct_Q8_0_gguf = "Phi-3-mini-128k-instruct-Q8_0.gguf"
    if os.path.exists(mistral_7b_instruct_v0_2_q8_0_llamafile):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    elif os.path.exists(Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    elif os.path.exists(mistral_7b_instruct_v0_2_q8_0_llamafile):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    else:
        logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
        print("Downloading LLM from Huggingface...")
        time.sleep(1)
        print("Gonna be a bit...")
        time.sleep(1)
        print("Like seriously, an 8GB file...")
        time.sleep(2)
        dl_check = input("Final chance to back out, hit 'N'/'n' to cancel, or 'Y'/'y' to continue: ")
        if dl_check == "N" or dl_check == "n":
            exit()
        else:
            print("Downloading LLM from Huggingface...")
            # Establish hash values for LLM models
            mistral_7b_instruct_v0_2_q8_gguf_sha256 = "f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06"
            samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
            mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"

            # FIXME - llm_choice
            llm_choice = 2
            llm_choice = input("Which LLM model would you like to download? 1. Mistral-7B-Instruct-v0.2-GGUF or 2. Samantha-Mistral-Instruct-7B-Bulleted-Notes) (plain or 'custom') or MS Flavor: Phi-3-mini-128k-instruct-Q8_0.gguf  \n\n\tPress '1' or '2' or '3' to specify: ")
            while llm_choice != "1" and llm_choice != "2" and llm_choice != "3":
                print("Invalid choice. Please try again.")
            if llm_choice == "1":
                llm_download_model = "Mistral-7B-Instruct-v0.2-Q8.llamafile"
                mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
                llm_download_model_hash = mistral_7b_instruct_v0_2_q8_0_llamafile_sha256
                llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
                llamafile_llm_output_filename = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "2":
                llm_download_model = "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf"
                samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
                llm_download_model_hash = samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256
                llamafile_llm_output_filename = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes_GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "3":
                llm_download_model = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                Phi_3_mini_128k_instruct_Q8_0_gguf_sha256 = "6817b66d1c3c59ab06822e9732f0e594eea44e64cae2110906eac9d17f75d193"
                llm_download_model_hash = Phi_3_mini_128k_instruct_Q8_0_gguf_sha256
                llamafile_llm_output_filename = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/gaianet/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct-Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "4": # FIXME - and meta_Llama_3_8B_Instruct_Q8_0_llamafile_exists == False:
                meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256 = "406868a97f02f57183716c7e4441d427f223fdbc7fa42964ef10c4d60dd8ed37"
                llm_download_model_hash = meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256
                llamafile_llm_output_filename = " Meta-Llama-3-8B-Instruct.Q8_0.llamafile"
                llamafile_llm_url = "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.llamafile?download=true"
            else:
                print("Invalid choice. Please try again.")
    return output_filename




# FIXME / IMPLEMENT FULLY
# File download verification
#mistral_7b_llamafile_instruct_v02_q8_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
#global mistral_7b_instruct_v0_2_q8_0_llamafile_sha256
#mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"

#mistral_7b_v02_instruct_model_q8_gguf_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf?download=true"
#global mistral_7b_instruct_v0_2_q8_gguf_sha256
#mistral_7b_instruct_v0_2_q8_gguf_sha256 = "f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06"

#samantha_instruct_model_q8_gguf_url = "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes_GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true"
#global samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256
#samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"


process = None
# Function to close out llamafile process on script exit.
def cleanup_process():
    global process
    if process is not None:
        process.kill()
        logging.debug("Main: Terminated the external process")


def signal_handler(sig, frame):
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)


# FIXME - Add callout to gradio UI
def local_llm_function():
    global process
    repo = "Mozilla-Ocho/llamafile"
    asset_name_prefix = "llamafile-"
    useros = os.name
    if useros == "nt":
        output_filename = "llamafile.exe"
    else:
        output_filename = "llamafile"
    print(
        "WARNING - Checking for existence of llamafile and HuggingFace model, downloading if needed...This could be a while")
    print("WARNING - and I mean a while. We're talking an 8 Gigabyte model here...")
    print("WARNING - Hope you're comfy. Or it's already downloaded.")
    time.sleep(6)
    logging.debug("Main: Checking and downloading Llamafile from Github if needed...")
    llamafile_path = download_latest_llamafile(repo, asset_name_prefix, output_filename)
    logging.debug("Main: Llamafile downloaded successfully.")

    # FIXME - llm_choice
    global llm_choice
    llm_choice = 1
    # Launch the llamafile in an external process with the specified argument
    if llm_choice == 1:
        arguments = ["--ctx-size", "8192 ", " -m", "mistral-7b-instruct-v0.2.Q8_0.llamafile"]
    elif llm_choice == 2:
        arguments = ["--ctx-size", "8192 ", " -m", "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"]
    elif llm_choice == 3:
        arguments = ["--ctx-size", "8192 ", " -m", "Phi-3-mini-128k-instruct-Q8_0.gguf"]
    elif llm_choice == 4:
        arguments = ["--ctx-size", "8192 ", " -m", "llama-3"] # FIXME

    try:
        logging.info("Main: Launching the LLM (llamafile) in an external terminal window...")
        if useros == "nt":
            launch_in_new_terminal_windows(llamafile_path, arguments)
        elif useros == "posix":
            launch_in_new_terminal_linux(llamafile_path, arguments)
        else:
            launch_in_new_terminal_mac(llamafile_path, arguments)
        # FIXME - pid doesn't exist in this context
        #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
        atexit.register(cleanup_process, process)
    except Exception as e:
        logging.error(f"Failed to launch the process: {e}")
        print(f"Failed to launch the process: {e}")


# This function is used to dl a llamafile binary + the Samantha Mistral Finetune model.
# It should only be called when the user is using the GUI to set up and interact with Llamafile.
def local_llm_gui_function(am_noob, verbose_checked, threads_checked, threads_value, http_threads_checked, http_threads_value,
                 model_checked, model_value, hf_repo_checked, hf_repo_value, hf_file_checked, hf_file_value,
                 ctx_size_checked, ctx_size_value, ngl_checked, ngl_value, host_checked, host_value, port_checked,
                 port_value):
    # Identify running OS
    useros = os.name
    if useros == "nt":
        output_filename = "llamafile.exe"
    else:
        output_filename = "llamafile"

    # Build up the commands for llamafile
    built_up_args = []

    # Identify if the user wants us to do everything for them
    if am_noob == True:
        print("You're a noob. (lol j/k; they're good settings)")

        # Setup variables for Model download from HF
        repo = "Mozilla-Ocho/llamafile"
        asset_name_prefix = "llamafile-"
        print(
            "WARNING - Checking for existence of llamafile or HuggingFace model (GGUF type), downloading if needed...This could be a while")
        print("WARNING - and I mean a while. We're talking an 8 Gigabyte model here...")
        print("WARNING - Hope you're comfy. Or it's already downloaded.")
        time.sleep(6)
        logging.debug("Main: Checking for Llamafile and downloading  from Github if needed...\n\tAlso checking for a "
                      "local LLM model...\n\tDownloading if needed...\n\tThis could take a while...\n\tWill be the "
                      "'samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf' model...")
        llamafile_path = download_latest_llamafile_through_gui(repo, asset_name_prefix, output_filename)
        logging.debug("Main: Llamafile downloaded successfully.")

        arguments = []
        # FIXME - llm_choice
        # This is the gui, we can add this as options later
        llm_choice = 2
        # Launch the llamafile in an external process with the specified argument
        if llm_choice == 1:
            arguments = ["--ctx-size", "8192 ", " -m", "mistral-7b-instruct-v0.2.Q8_0.llamafile"]
        elif llm_choice == 2:
            arguments = """--ctx-size 8192 -m samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"""
        elif llm_choice == 3:
            arguments = ["--ctx-size", "8192 ", " -m", "Phi-3-mini-128k-instruct-Q8_0.gguf"]
        elif llm_choice == 4:
            arguments = ["--ctx-size", "8192 ", " -m", "llama-3"]

        try:
            logging.info("Main(Local-LLM-GUI-noob): Launching the LLM (llamafile) in an external terminal window...")

            if useros == "nt":
                command = 'start cmd /k "llamafile.exe --ctx-size 8192 -m samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"'
                subprocess.Popen(command, shell=True)
            elif useros == "posix":
                command = "llamafile --ctx-size 8192 -m samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
                subprocess.Popen(command, shell=True)
            else:
                command = "llamafile.exe --ctx-size 8192 -m samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
                subprocess.Popen(command, shell=True)
            # FIXME - pid doesn't exist in this context
            # logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
            atexit.register(cleanup_process, process)
        except Exception as e:
            logging.error(f"Failed to launch the process: {e}")
            print(f"Failed to launch the process: {e}")

    else:
        print("You're not a noob.")
        llamafile_path = download_latest_llamafile_no_model(output_filename)
        if verbose_checked == True:
            print("Verbose mode enabled.")
            built_up_args.append("--verbose")
        if threads_checked == True:
            print(f"Threads enabled with value: {threads_value}")
            built_up_args.append(f"--threads {threads_value}")
        if http_threads_checked == True:
            print(f"HTTP Threads enabled with value: {http_threads_value}")
            built_up_args.append(f"--http-threads {http_threads_value}")
        if model_checked == True:
            print(f"Model enabled with value: {model_value}")
            built_up_args.append(f"--model {model_value}")
        if hf_repo_checked == True:
            print(f"Huggingface repo enabled with value: {hf_repo_value}")
            built_up_args.append(f"--hf-repo {hf_repo_value}")
        if hf_file_checked == True:
            print(f"Huggingface file enabled with value: {hf_file_value}")
            built_up_args.append(f"--hf-file {hf_file_value}")
        if ctx_size_checked == True:
            print(f"Context size enabled with value: {ctx_size_value}")
            built_up_args.append(f"--ctx-size {ctx_size_value}")
        if ngl_checked == True:
            print(f"NGL enabled with value: {ngl_value}")
            built_up_args.append(f"--ngl {ngl_value}")
        if host_checked == True:
            print(f"Host enabled with value: {host_value}")
            built_up_args.append(f"--host {host_value}")
        if port_checked == True:
            print(f"Port enabled with value: {port_value}")
            built_up_args.append(f"--port {port_value}")

        # Lets go ahead and finally launch the bastard...
        try:
            logging.info("Main(Local-LLM-GUI-Main): Launching the LLM (llamafile) in an external terminal window...")
            if useros == "nt":
                launch_in_new_terminal_windows(llamafile_path, built_up_args)
            elif useros == "posix":
                launch_in_new_terminal_linux(llamafile_path, built_up_args)
            else:
                launch_in_new_terminal_mac(llamafile_path, built_up_args)
            # FIXME - pid doesn't exist in this context
            #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
            atexit.register(cleanup_process, process)
        except Exception as e:
            logging.error(f"Failed to launch the process: {e}")
            print(f"Failed to launch the process: {e}")


# Launch the executable in a new terminal window # FIXME - really should figure out a cleaner way of doing this...
def launch_in_new_terminal_windows(executable, args):
    command = f'start cmd /k "{executable} {" ".join(args)}"'
    subprocess.Popen(command, shell=True)


# FIXME
def launch_in_new_terminal_linux(executable, args):
    command = f'gnome-terminal -- {executable} {" ".join(args)}'
    subprocess.Popen(command, shell=True)


# FIXME
def launch_in_new_terminal_mac(executable, args):
    command = f'open -a Terminal.app {executable} {" ".join(args)}'
    subprocess.Popen(command, shell=True)

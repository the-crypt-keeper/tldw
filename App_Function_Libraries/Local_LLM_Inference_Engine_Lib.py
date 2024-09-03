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
#import atexit
import re
import subprocess
import sys
import time

from App_Function_Libraries.Utils.Utils import download_file
# Import 3rd-pary Libraries
#
# Import Local
from Article_Summarization_Lib import *

#
#
#######################################################################################################################
# Function Definitions
#


# Function to download the latest llamafile from the Mozilla-Ocho/llamafile repo
def download_latest_llamafile(output_filename):
    # Check if the file already exists
    print("Checking for and downloading Llamafile it it doesn't already exist...")
    if os.path.exists(output_filename):
        print("Llamafile already exists. Skipping download.")
        logging.debug(f"{output_filename} already exists. Skipping download.")
        llamafile_exists = True
    else:
        llamafile_exists = False
    # Double check if the file exists
    if llamafile_exists:
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


def download_llm_model(model_name, model_url, model_filename, model_hash):
    print("Checking available LLM models:")
    available_models = []
    missing_models = []

    for key, model in llm_models.items():
        if os.path.exists(model['filename']):
            print(f"{key}. {model['name']} (Available)")
            available_models.append(key)
        else:
            print(f"{key}. {model['name']} (Not downloaded)")
            missing_models.append(key)

    if not available_models:
        print("No models are currently downloaded.")
    else:
        print(f"\n{len(available_models)} model(s) are available for use.")

    action = input("Do you want to (u)se an available model, (d)ownload a new model, or (q)uit? ").lower()

    if action == 'u':
        if not available_models:
            print("No models are available. Please download a model first.")
            return None
        while True:
            choice = input(f"Enter the number of the model you want to use ({', '.join(available_models)}): ")
            if choice in available_models:
                print(f"Selected model: {llm_models[choice]['name']}")
                return llm_models[choice]['filename']
            else:
                print("Invalid choice. Please try again.")

    elif action == 'd':
        if not missing_models:
            print("All models are already downloaded. You can use an available model.")
            return None
        print("\nThe following models can be downloaded:")
        for key in missing_models:
            print(f"{key}. {llm_models[key]['name']}")
        while True:
            choice = input(f"Enter the number of the model you want to download ({', '.join(missing_models)}): ")
            if choice in missing_models:
                model = llm_models[choice]
                print(f"Downloading {model['name']}...")
                download_file(model['url'], model['filename'], expected_checksum=model['hash'])
                print(f"{model['filename']} has been downloaded successfully.")
                return model['filename']
            else:
                print("Invalid choice. Please try again.")

    elif action == 'q':
        print("Exiting model selection.")
        return None

    else:
        print("Invalid action. Exiting model selection.")
        return None






#
#
########################################
#
# LLM models information


llm_models = {
    "1": {
        "name": "Mistral-7B-Instruct-v0.2-Q8.llamafile",
        "url": "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true",
        "filename": "mistral-7b-instruct-v0.2.Q8_0.llamafile",
        "hash": "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
    },
    "2": {
        "name": "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf",
        "url": "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes-GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true",
        "filename": "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf",
        "hash": "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
    },
    "3": {
        "name": "Phi-3-mini-128k-instruct-Q8_0.gguf",
        "url": "https://huggingface.co/gaianet/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct-Q8_0.gguf?download=true",
        "filename": "Phi-3-mini-128k-instruct-Q8_0.gguf",
        "hash": "6817b66d1c3c59ab06822e9732f0e594eea44e64cae2110906eac9d17f75d193"
    },
    "4": {
        "name": "Meta-Llama-3-8B-Instruct.Q8_0.llamafile",
        "url": "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.llamafile?download=true",
        "filename": "Meta-Llama-3-8B-Instruct.Q8_0.llamafile",
        "hash": "406868a97f02f57183716c7e4441d427f223fdbc7fa42964ef10c4d60dd8ed37"
    }
}


process = None
# Function to close out llamafile process on script exit.
def cleanup_process():
    global process
    if process is not None:
        # FIXME - process.kill()
        #process.kill()
        logging.debug("Main: Terminated the external process")


def signal_handler(sig, frame):
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)


# FIXME - Add callout to gradio UI
def local_llm_function():
    global process
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
    llamafile_path = download_latest_llamafile(output_filename)
    logging.debug("Main: Llamafile downloaded successfully.")

    # FIXME - llm_choice
    input("What model do you want to use? (Press Enter to continue)")
    print("1. Mistral-7B-Instruct-v0.2-Q8.llamafile")
    print("2. Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf")
    print("3. Phi-3-mini-128k-instruct-Q8_0.gguf")
    print("4. Meta-Llama-3-8B-Instruct.Q8_0.llamafile")
    llm_choice = int(input("Enter the number of the model you want to use: "))
    if llm_choice not in [1, 2, 3, 4]:
        print("Invalid choice. Exiting.")
        return
    arguments = []
    # Launch the llamafile in an external process with the specified argument
    if llm_choice == 1:
        arguments = ["--ctx-size", "8192 ", " -m", "mistral-7b-instruct-v0.2.Q8_0.llamafile"]
    elif llm_choice == 2:
        arguments = ["--ctx-size", "8192 ", " -m", "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"]
    elif llm_choice == 3:
        arguments = ["--ctx-size", "8192 ", " -m", "Phi-3-mini-128k-instruct-Q8_0.gguf"]
    elif llm_choice == 4:
        arguments = ["--ctx-size", "8192 ", " -m", "Meta-Llama-3-8B-Instruct.Q8_0.llamafile"] # FIXME

    try:
        logging.info("local_llm_function: Launching the LLM (llamafile) in an external terminal window...")
        if useros == "nt":
            launch_in_new_terminal_windows(llamafile_path, arguments)
        elif useros == "posix":
            launch_in_new_terminal_linux(llamafile_path, arguments)
        else:
            launch_in_new_terminal_mac(llamafile_path, arguments)
        # FIXME - pid doesn't exist in this context
        #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
        # Ha like this shit works
        #atexit.register(cleanup_process, process)
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
    if am_noob:
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
        llamafile_path = download_latest_llamafile(output_filename)
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
            arguments = ["--ctx-size", "8192 ", " -m", "Meta-Llama-3-8B-Instruct.Q8_0.llamafile"]

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
            #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
            # FIXME - Shit just don't work
            # atexit.register(cleanup_process, process)
        except Exception as e:
            logging.error(f"Failed to launch the process: {e}")
            print(f"Failed to launch the process: {e}")

    else:
        print("You're not a noob.")
        llamafile_path = download_latest_llamafile(output_filename)
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
            # FIXME
            #atexit.register(cleanup_process, process)
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

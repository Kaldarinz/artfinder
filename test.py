import aiohttp
import asyncio
import pandas as pd
import os

async def download_file(url: str, save_path: str, task_id: int, progress_list: list):
    """
    Download a file from the given URL and save it to the specified path, with progress printing.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    save_path : str
        The local path where the file will be saved.
    task_id : int
        The ID of the task for identifying progress.
    progress_list : list
        A shared list to store progress for each task.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                total_size = int(response.headers.get('Content-Length', 0))  # Get total file size
                downloaded_size = 0

                with open(save_path, 'wb') as f:
                    while chunk := await response.content.read(1024):  # Read in chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Update progress for this task
                        progress = (downloaded_size / total_size) * 100 if total_size else 0
                        progress_list[task_id] = f"Task {task_id}: Downloading: {progress:.2f}% ({downloaded_size/1024:.1f}/{total_size/1024:.1f} kb)"
                        print_progress(progress_list)

                progress_list[task_id] = f"Task {task_id}: File downloaded: {save_path}"
                print_progress(progress_list)
            else:
                progress_list[task_id] = f"Task {task_id}: Failed to download file. HTTP status: {response.status}"
                print_progress(progress_list)

def print_progress(progress_list):
    """
    Print the progress of all tasks on separate lines.

    Parameters
    ----------
    progress_list : list
        A list containing progress messages for all tasks.
    """
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
    for progress in progress_list:
        print(progress)

async def main():
    save_paths = ["file1.pdf", "file2.pdf"]
    full_texts = pd.read_csv("full_text_links.csv").values.tolist()

    # Example URLs for demonstration
    urls = [full_texts[1][0], full_texts[2][0]]

    # Initialize a shared progress list
    progress_list = [""] * len(urls)

    # Run multiple download tasks
    tasks = [
        download_file(url, save_path, task_id, progress_list)
        for task_id, (url, save_path) in enumerate(zip(urls, save_paths))
    ]
    await asyncio.gather(*tasks)

# Run the main coroutine
asyncio.run(main())
import os
import subprocess

def run_cmd(file_name):

    command = [
        "litmus7",
        "-mach", "/home/whq/Desktop/slide/tests/litmus/riscv.cfg",  # 指定配置文件
        "-avail", "4",
        file_name,
        "-o", "SB"
    ]

    try:

        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


        print("Command executed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:

        print("Command failed with error:")
        print(e.stderr)

def create_folders_for_litmus_files(directory):

    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    

    for root, dirs, files in os.walk(directory):
        for filename in files:

            if filename.endswith('.litmus'):
                folder_name = os.path.splitext(filename)[0]
                new_folder_path = os.path.join(directory, folder_name)
                

                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                    print(f"Created folder: {new_folder_path}")
                else:
                    print(f"Folder '{new_folder_path}' already exists.")
            else:
                print(f"'{filename}' is not a .litmus file, skipping.")

        # run cmd
        run_cmd(filename)



directory_path = "/home/whq/Desktop/slide/tests/input/litmus/manual"



if __name__ == '__main__':
    create_folders_for_litmus_files(directory_path)
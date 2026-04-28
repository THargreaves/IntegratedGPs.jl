module DOWNLOAD_DATASETS

using Base: SecretBuffer

using HTTP
using JSON3: JSON3
using ZipFile

using ProgressMeter
using Crayons

download_target_path = "teamtrack.zip"
football_path = ["teamtrack", "teamtrack", "soccer_top"]

dataset_url = "https://www.kaggle.com/api/v1/datasets/download/atomscott/teamtrack"

if !isfile(download_target_path)
    cred_path = joinpath(homedir(), ".kaggle", "kaggle.json")
    if isfile(cred_path)
        println(
            "Reading credentials at $(Crayon(foreground=:yellow))$cred_path.$(Crayon(reset=true))",
        )
        creds = JSON3.read(cred_path)
        kaggle_user = creds["username"]
        kaggle_key = creds["key"]
    else
        println(
            "Could not find kaggle credentials file at $(Crayon(foreground=:red))$cred_path$(Crayon(reset=true)).",
        )
        println("Please provide your kaggle username and API key manually.")
        print("Kaggle username: ")
        kaggle_user = readline()

        key_buf = Base.getpass("Kaggle API key")
        kaggle_key = read(key_buf, String)
        Base.shred!(key_buf)
        println("")
    end

    println(
        "Download might take several minutes: $(Crayon(foreground=:yellow))~17GB$(reset=true).",
    )
    p = ProgressUnknown("Downloading dataset..."; spinner=true)
    t = @async HTTP.get(dataset_url, auth=(kaggle_user, kaggle_key))

    while !istaskdone(t)
        next!(p)
        sleep(0.5)
    end

    finish!(p)
    resp = fetch(t)
    if resp.status == 200
        println("Downloaded successfully.")
        write(download_target_path, resp.body)
        println("Write successful.")
    else
        println("Error: ", String(resp.body))
    end
else
    println(
        "Found $(Crayon(foreground=:green))$download_target_path$(Crayon(reset=true)). Skipping download.",
    )
end

r = ZipFile.Reader(download_target_path)

teamtrack_inds = findall(
    f -> all(splitpath(f.name)[i] == football_path[i] for i in eachindex(football_path)),
    r.files,
)
filtered_files = r.files[teamtrack_inds]

max_filename_buffer_length = 10
filename_buffer = []

p = Progress(length(filtered_files); desc="Extracting $download_target_path...")

for (ind, f) in enumerate(filtered_files)
    out_path = f.name
    dir = dirname(out_path)
    split_dir = splitpath(dir)
    mkpath(dir)
    if !endswith(f.name, "/") && !isfile(out_path)
        write(out_path, read(f))
    end

    if length(filename_buffer) == max_filename_buffer_length
        deleteat!(filename_buffer, 1)
    end
    push!(filename_buffer, out_path)

    next!(
        p;
        showvalues=[
            (ind - length(filename_buffer) + prev_ind, prev) for
            (prev_ind, prev) in enumerate(filename_buffer)
        ],
    )
end
close(r)

println("Done.")
end
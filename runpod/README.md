Useful links:

Runpod sdk docs: https://docs.runpod.io/sdks/python/overview

Install runpodctl (linux):
```
wget -qO- cli.runpod.net | sudo bash
```

Dockerhub with images: https://hub.docker.com/repository/docker/r1vlm/r1_vlm/general

How to build the image:
```
./build_docker.sh
```

How to push the image (requires `docker login` first). The script automatically versions it:
```
./build_and_push.sh r1vlm
```

Once you've created a pod, go to https://www.runpod.io/console/pods and find the pod you just created. Click "Connect" and you'll see the command you need to run to connect to the pod. Adjust the path to the ssh keys as needed.

```
ssh <runpod pod id>@ssh.runpod.io -i ~/.ssh/runpod.pem

# or

ssh root@<pod ip> -p <port> -i ~/.ssh/runpod.pem
```

To set up vscode, add the following to your ssh config:
```
Host <your_pod_instance>
    HostName <pod ip>
    User root
    Port <port>
    IdentityFile ~/.ssh/runpod.pem
```

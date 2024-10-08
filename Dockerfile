FROM ubuntu:latest
LABEL authors="titus"

ENTRYPOINT ["top", "-b"]
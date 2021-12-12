FROM Windows

RUN apt-get update
RUN apt-get install python

CMD ["/src/main.py"]
ENTRYPOINT ["python"]
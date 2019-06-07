# get R
FROM r-base:3.6.0

# copy files
COPY . /usr/local/src/
WORKDIR /usr/local/src/

# install packages
RUN Rscript /usr/local/src/install_packages.R

# define entrypoint
ENTRYPOINT ["Rscript", "predict_tweet.R"]
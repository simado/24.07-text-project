FROM python:3.11-slim-buster
LABEL maintainer="Simonas Adomavicius"
LABEL project="text classification"

# Never prompt the user for choices on installation/configuration of packages
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux
# Define en_US.
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

# main variables:
ENV PROJECT_DIR=/opt/app

# define dependencies
ENV buildDeps=' \
    locales \
    wget \
    zip \
    unzip \
'

# SETUP container environment
RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends ${buildDeps} \
    && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Copying relevant files to their location in airflow
COPY ./app $PROJECT_DIR/app
COPY ./data $PROJECT_DIR/data
COPY ./requirements.in /requirements.in
COPY ./entrypoint.sh /entrypoint.sh

# Install Python packages
RUN set -ex \
    && pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir -r /requirements.in

# download nltk data
RUN set -ex \
    && python -m nltk.downloader stopwords \
    && python -m nltk.downloader wordnet

# clean after installation to reduce image size
RUN set -ex \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base


WORKDIR ${PROJECT_DIR}
EXPOSE 80
ENTRYPOINT [ "/entrypoint.sh" ]

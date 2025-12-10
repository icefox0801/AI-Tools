#!/bin/bash
# ================================================
# Configure APT mirrors for China (Ubuntu/Debian)
# ================================================
# This script should be called from Dockerfile with:
#   COPY --from=scripts setup-china-mirrors.sh /tmp/
#   RUN /tmp/setup-china-mirrors.sh && rm /tmp/setup-china-mirrors.sh
#
# Or inline in Dockerfile RUN commands

# Detect OS and version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_CODENAME
else
    echo "Cannot detect OS, skipping mirror configuration"
    exit 0
fi

case $OS in
    ubuntu)
        echo "Configuring Aliyun mirror for Ubuntu $VERSION"
        cat > /etc/apt/sources.list << EOF
deb https://mirrors.aliyun.com/ubuntu/ $VERSION main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ $VERSION-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ $VERSION-security main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ $VERSION-backports main restricted universe multiverse
EOF
        ;;
    debian)
        echo "Configuring Aliyun mirror for Debian $VERSION"
        cat > /etc/apt/sources.list << EOF
deb https://mirrors.aliyun.com/debian/ $VERSION main contrib non-free
deb https://mirrors.aliyun.com/debian/ $VERSION-updates main contrib non-free
deb https://mirrors.aliyun.com/debian-security/ $VERSION-security main contrib non-free
EOF
        ;;
    *)
        echo "Unknown OS: $OS, skipping mirror configuration"
        ;;
esac

echo "Mirror configuration complete"

#ifndef IPPL_BASE_STREAM_H
#define IPPL_BASE_STREAM_H

namespace ippl {

    template <class Object>
    class BaseStream {
    public:

        virtual void open(std::string filename) = 0;

        virtual void close() = 0;

        virtual void operator<<(const Object& obj) = 0;

        virtual void operator>>(Object& obj) = 0;
    };
}  // namespace ippl

#endif
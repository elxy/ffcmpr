/*
 * Copyright (c) 2003 Fabrice Bellard
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * simple media player based on the FFmpeg libraries
 */

#include "config.h"
#include "config_components.h"

#include <assert.h>
#include <math.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>

#include "libavutil/avstring.h"
#include "libavutil/mathematics.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"
#include "libavutil/dict.h"
#include "libavutil/fifo.h"
#include "libavutil/imgutils.h"
#include "libavutil/avassert.h"
#include "libavutil/time.h"
#include "libavutil/bprint.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswscale/swscale.h"
#include "libavutil/opt.h"
#include "libavutil/tx.h"

#include "libavfilter/avfilter.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"

#include <SDL.h>
#include <SDL_thread.h>

#include "cmdutils.h"
#include "ffplay_renderer.h"
#include "opt_common.h"

const char program_name[] = "ffcmpr";
const int program_birth_year = 2020;

#define MAX_QUEUE_SIZE (15 * 1024 * 1024)
#define MIN_FRAMES 25
#define EXTERNAL_CLOCK_MIN_FRAMES 2
#define EXTERNAL_CLOCK_MAX_FRAMES 10

/* no AV sync correction is done if below the minimum AV sync threshold */
#define AV_SYNC_THRESHOLD_MIN 0.04
/* AV sync correction is done if above the maximum AV sync threshold */
#define AV_SYNC_THRESHOLD_MAX 0.1
/* If a frame duration is longer than this, it will not be duplicated to compensate AV sync */
#define AV_SYNC_FRAMEDUP_THRESHOLD 0.1
/* no AV correction is done if too big error */
#define AV_NOSYNC_THRESHOLD 10.0

/* external clock speed adjustment constants for realtime sources based on buffer fullness */
#define EXTERNAL_CLOCK_SPEED_MIN  0.900
#define EXTERNAL_CLOCK_SPEED_MAX  1.010
#define EXTERNAL_CLOCK_SPEED_STEP 0.001

/* polls for possible required screen refresh at least this often, should be less than 1/fps */
#define REFRESH_RATE 0.01

typedef struct MyAVPacketList {
    AVPacket *pkt;
    int serial;
} MyAVPacketList;

typedef struct PacketQueue {
    AVFifo *pkt_list;
    int nb_packets;
    int size;
    int64_t duration;
    int abort_request;
    int serial;
    SDL_mutex *mutex;
    SDL_cond *cond;
} PacketQueue;

#define VIDEO_PICTURE_QUEUE_SIZE 3
#define FRAME_QUEUE_SIZE VIDEO_PICTURE_QUEUE_SIZE

typedef struct Clock {
    double pts;           /* clock base */
    double pts_drift;     /* clock base minus time at which we updated the clock */
    double last_updated;
    double speed;
    int serial;           /* clock is based on a packet with this serial */
    int paused;
    int *queue_serial;    /* pointer to the current packet queue serial, used for obsolete clock detection */
} Clock;

typedef struct FrameData {
    int64_t pkt_pos;
} FrameData;

/* Common struct for handling all types of decoded data and allocated render buffers. */
typedef struct Frame {
    AVFrame *frame;
    int serial;
    double pts;           /* presentation timestamp for the frame */
    double duration;      /* estimated duration of the frame */
    int64_t pos;          /* byte position of the frame in the input file */
    int width;
    int height;
    int format;
    AVRational sar;
    int uploaded;
    int flip_v;
} Frame;

typedef struct FrameQueue {
    Frame queue[FRAME_QUEUE_SIZE];
    int rindex;
    int windex;
    int size;
    int max_size;
    int keep_last;
    int rindex_shown;
    SDL_mutex *mutex;
    SDL_cond *cond;
    PacketQueue *pktq;
} FrameQueue;

enum {
    AV_SYNC_VIDEO_MASTER,
    AV_SYNC_EXTERNAL_CLOCK, /* synchronize to an external clock */
};

typedef struct Decoder {
    AVPacket *pkt;
    PacketQueue *queue;
    AVCodecContext *avctx;
    int pkt_serial;
    int finished;
    int packet_pending;
    SDL_cond *empty_queue_cond;
    int64_t start_pts;
    AVRational start_pts_tb;
    int64_t next_pts;
    AVRational next_pts_tb;
    SDL_Thread *decoder_tid;
} Decoder;

typedef struct VideoState {
    SDL_Thread *read_tid;
    const AVInputFormat *iformat;
    int abort_request;
    int force_refresh;
    int paused;
    int last_paused;
    int queue_attachments_req;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;
    int64_t seek_rel;
    int read_pause_return;
    AVFormatContext *ic;
    int realtime;

    Clock vidclk;
    Clock extclk;

    FrameQueue pictq;

    Decoder viddec;

    int av_sync_type;

    struct SwrContext *swr_ctx;
    int frame_drops_early;
    int frame_drops_late;

    SDL_Texture *vis_texture;
    SDL_Texture *vid_texture;
    uint32_t *texture_buffer;

    double frame_timer;
    double frame_last_returned_time;
    double frame_last_filter_delay;
    int video_stream;
    AVStream *video_st;
    PacketQueue videoq;
    double max_frame_duration;      // maximum duration of a frame - above this, we consider the jump a timestamp discontinuity
    int eof;

    char *filename;
    int width, height, xleft, ytop;
    int step;

    int vfilter_idx;
    AVFilterContext *in_video_filter;   // the first filter in the video chain
    AVFilterContext *out_video_filter;  // the last filter in the video chain

    int last_video_stream;

    SDL_cond *continue_read_thread;

    struct AVDictionary *format_opts;
    int index;
    int window_id;
    const char *window_title;
    double zoom_ratio;
    int window_attached_mode;  // if true, resize with the window
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_RendererInfo renderer_info;
    VkRenderer *vk_renderer;
    int frame_queued;
    int frame_displayed;
} VideoState;

#define MAX_INPUT_NUM (16)

/* options specified by the user */
static const AVInputFormat *file_iformat;
static int nb_inputs = 0;
static const char *input_filenames[MAX_INPUT_NUM];
static VideoState *input_vs[MAX_INPUT_NUM];
static VideoState *master_vs = NULL;
static int default_width  = 640;
static int default_height = 480;
static int screen_width  = 0;
static int screen_height = 0;
static int screen_left = SDL_WINDOWPOS_CENTERED;
static int screen_top = SDL_WINDOWPOS_CENTERED;
static const char* wanted_stream_spec[AVMEDIA_TYPE_NB] = {0};
static int seek_by_bytes = -1;
static float seek_interval = 1;
static int borderless;
static int alwaysontop;
static int show_status = 1;
static int av_sync_type = AV_SYNC_VIDEO_MASTER;
static int64_t start_time = AV_NOPTS_VALUE;
static int fast = 0;
static int genpts = 0;
static int lowres = 0;
static int decoder_reorder_pts = -1;
static int autoexit;
static int loop = 1;
static int framedrop = -1;
static int infinite_buffer = -1;
static const char *video_codec_name;
static const char **vfilters_list = NULL;
static int nb_vfilters = 0;
static int autorotate = 1;
static int find_stream_info = 1;
static int filter_nbthreads = 0;
static int enable_vulkan = 0;
static char *vulkan_params = NULL;
static const char *hwaccel = NULL;

static int use_10bit = -1;
static int no_colorspace_hint = 0;
static const char *save_format = NULL;

/* current context */
static int is_full_screen;

#define FF_QUIT_EVENT    (SDL_USEREVENT + 2)

static const struct TextureFormatEntry {
    enum AVPixelFormat format;
    int texture_fmt;
} sdl_texture_format_map[] = {
    { AV_PIX_FMT_RGB8,           SDL_PIXELFORMAT_RGB332 },
    { AV_PIX_FMT_RGB444,         SDL_PIXELFORMAT_RGB444 },
    { AV_PIX_FMT_RGB555,         SDL_PIXELFORMAT_RGB555 },
    { AV_PIX_FMT_BGR555,         SDL_PIXELFORMAT_BGR555 },
    { AV_PIX_FMT_RGB565,         SDL_PIXELFORMAT_RGB565 },
    { AV_PIX_FMT_BGR565,         SDL_PIXELFORMAT_BGR565 },
    { AV_PIX_FMT_RGB24,          SDL_PIXELFORMAT_RGB24 },
    { AV_PIX_FMT_BGR24,          SDL_PIXELFORMAT_BGR24 },
    { AV_PIX_FMT_0RGB32,         SDL_PIXELFORMAT_RGB888 },
    { AV_PIX_FMT_0BGR32,         SDL_PIXELFORMAT_BGR888 },
    { AV_PIX_FMT_NE(RGB0, 0BGR), SDL_PIXELFORMAT_RGBX8888 },
    { AV_PIX_FMT_NE(BGR0, 0RGB), SDL_PIXELFORMAT_BGRX8888 },
    { AV_PIX_FMT_RGB32,          SDL_PIXELFORMAT_ARGB8888 },
    { AV_PIX_FMT_RGB32_1,        SDL_PIXELFORMAT_RGBA8888 },
    { AV_PIX_FMT_BGR32,          SDL_PIXELFORMAT_ABGR8888 },
    { AV_PIX_FMT_BGR32_1,        SDL_PIXELFORMAT_BGRA8888 },
    { AV_PIX_FMT_YUV420P,        SDL_PIXELFORMAT_IYUV },
    { AV_PIX_FMT_YUYV422,        SDL_PIXELFORMAT_YUY2 },
    { AV_PIX_FMT_UYVY422,        SDL_PIXELFORMAT_UYVY },
    { AV_PIX_FMT_RGB48LE,        SDL_PIXELFORMAT_ARGB2101010 },
    { AV_PIX_FMT_NONE,           SDL_PIXELFORMAT_UNKNOWN },
};

static int opt_add_vfilter(void *optctx, const char *opt, const char *arg)
{
    int ret = GROW_ARRAY(vfilters_list, nb_vfilters);
    if (ret < 0)
        return ret;

    vfilters_list[nb_vfilters - 1] = av_strdup(arg);
    if (!vfilters_list[nb_vfilters - 1])
        return AVERROR(ENOMEM);

    return 0;
}

static int packet_queue_put_private(PacketQueue *q, AVPacket *pkt)
{
    MyAVPacketList pkt1;
    int ret;

    if (q->abort_request)
       return -1;


    pkt1.pkt = pkt;
    pkt1.serial = q->serial;

    ret = av_fifo_write(q->pkt_list, &pkt1, 1);
    if (ret < 0)
        return ret;
    q->nb_packets++;
    q->size += pkt1.pkt->size + sizeof(pkt1);
    q->duration += pkt1.pkt->duration;
    /* XXX: should duplicate packet data in DV case */
    SDL_CondSignal(q->cond);
    return 0;
}

static int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacket *pkt1;
    int ret;

    pkt1 = av_packet_alloc();
    if (!pkt1) {
        av_packet_unref(pkt);
        return -1;
    }
    av_packet_move_ref(pkt1, pkt);

    SDL_LockMutex(q->mutex);
    ret = packet_queue_put_private(q, pkt1);
    SDL_UnlockMutex(q->mutex);

    if (ret < 0)
        av_packet_free(&pkt1);

    return ret;
}

static int packet_queue_put_nullpacket(PacketQueue *q, AVPacket *pkt, int stream_index)
{
    pkt->stream_index = stream_index;
    return packet_queue_put(q, pkt);
}

/* packet queue handling */
static int packet_queue_init(PacketQueue *q)
{
    memset(q, 0, sizeof(PacketQueue));
    q->pkt_list = av_fifo_alloc2(1, sizeof(MyAVPacketList), AV_FIFO_FLAG_AUTO_GROW);
    if (!q->pkt_list)
        return AVERROR(ENOMEM);
    q->mutex = SDL_CreateMutex();
    if (!q->mutex) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        return AVERROR(ENOMEM);
    }
    q->cond = SDL_CreateCond();
    if (!q->cond) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
        return AVERROR(ENOMEM);
    }
    q->abort_request = 1;
    return 0;
}

static void packet_queue_flush(PacketQueue *q)
{
    MyAVPacketList pkt1;

    SDL_LockMutex(q->mutex);
    while (av_fifo_read(q->pkt_list, &pkt1, 1) >= 0)
        av_packet_free(&pkt1.pkt);
    q->nb_packets = 0;
    q->size = 0;
    q->duration = 0;
    q->serial++;
    SDL_UnlockMutex(q->mutex);
}

static void packet_queue_destroy(PacketQueue *q)
{
    packet_queue_flush(q);
    av_fifo_freep2(&q->pkt_list);
    SDL_DestroyMutex(q->mutex);
    SDL_DestroyCond(q->cond);
}

static void packet_queue_abort(PacketQueue *q)
{
    SDL_LockMutex(q->mutex);

    q->abort_request = 1;

    SDL_CondSignal(q->cond);

    SDL_UnlockMutex(q->mutex);
}

static void packet_queue_start(PacketQueue *q)
{
    SDL_LockMutex(q->mutex);
    q->abort_request = 0;
    q->serial++;
    SDL_UnlockMutex(q->mutex);
}

/* return < 0 if aborted, 0 if no packet and > 0 if packet.  */
static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block, int *serial)
{
    MyAVPacketList pkt1;
    int ret;

    SDL_LockMutex(q->mutex);

    for (;;) {
        if (q->abort_request) {
            ret = -1;
            break;
        }

        if (av_fifo_read(q->pkt_list, &pkt1, 1) >= 0) {
            q->nb_packets--;
            q->size -= pkt1.pkt->size + sizeof(pkt1);
            q->duration -= pkt1.pkt->duration;
            av_packet_move_ref(pkt, pkt1.pkt);
            if (serial)
                *serial = pkt1.serial;
            av_packet_free(&pkt1.pkt);
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            SDL_CondWait(q->cond, q->mutex);
        }
    }
    SDL_UnlockMutex(q->mutex);
    return ret;
}

static int decoder_init(Decoder *d, AVCodecContext *avctx, PacketQueue *queue, SDL_cond *empty_queue_cond) {
    memset(d, 0, sizeof(Decoder));
    d->pkt = av_packet_alloc();
    if (!d->pkt)
        return AVERROR(ENOMEM);
    d->avctx = avctx;
    d->queue = queue;
    d->empty_queue_cond = empty_queue_cond;
    d->start_pts = AV_NOPTS_VALUE;
    d->pkt_serial = -1;
    return 0;
}

static int decoder_decode_frame(Decoder *d, AVFrame *frame) {
    int ret = AVERROR(EAGAIN);

    for (;;) {
        if (d->queue->serial == d->pkt_serial) {
            do {
                if (d->queue->abort_request)
                    return -1;

                switch (d->avctx->codec_type) {
                    case AVMEDIA_TYPE_VIDEO:
                        ret = avcodec_receive_frame(d->avctx, frame);
                        if (ret >= 0) {
                            if (decoder_reorder_pts == -1) {
                                frame->pts = frame->best_effort_timestamp;
                            } else if (!decoder_reorder_pts) {
                                frame->pts = frame->pkt_dts;
                            }
                        }
                    default:
                        break;
                }
                if (ret == AVERROR_EOF) {
                    d->finished = d->pkt_serial;
                    avcodec_flush_buffers(d->avctx);
                    return 0;
                }
                if (ret >= 0)
                    return 1;
            } while (ret != AVERROR(EAGAIN));
        }

        do {
            if (d->queue->nb_packets == 0)
                SDL_CondSignal(d->empty_queue_cond);
            if (d->packet_pending) {
                d->packet_pending = 0;
            } else {
                int old_serial = d->pkt_serial;
                if (packet_queue_get(d->queue, d->pkt, 1, &d->pkt_serial) < 0)
                    return -1;
                if (old_serial != d->pkt_serial) {
                    avcodec_flush_buffers(d->avctx);
                    d->finished = 0;
                    d->next_pts = d->start_pts;
                    d->next_pts_tb = d->start_pts_tb;
                }
            }
            if (d->queue->serial == d->pkt_serial)
                break;
            av_packet_unref(d->pkt);
        } while (1);

        if (d->pkt->buf && !d->pkt->opaque_ref) {
            FrameData *fd;

            d->pkt->opaque_ref = av_buffer_allocz(sizeof(*fd));
            if (!d->pkt->opaque_ref)
                return AVERROR(ENOMEM);
            fd = (FrameData*)d->pkt->opaque_ref->data;
            fd->pkt_pos = d->pkt->pos;
        }

        if (avcodec_send_packet(d->avctx, d->pkt) == AVERROR(EAGAIN)) {
            av_log(d->avctx, AV_LOG_ERROR, "Receive_frame and send_packet both returned EAGAIN, which is an API violation.\n");
            d->packet_pending = 1;
        } else {
            av_packet_unref(d->pkt);
        }
    }
}

static void decoder_destroy(Decoder *d) {
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->avctx);
}

static void frame_queue_unref_item(Frame *vp)
{
    av_frame_unref(vp->frame);
}

static int frame_queue_init(FrameQueue *f, PacketQueue *pktq, int max_size, int keep_last)
{
    int i;
    memset(f, 0, sizeof(FrameQueue));
    if (!(f->mutex = SDL_CreateMutex())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        return AVERROR(ENOMEM);
    }
    if (!(f->cond = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
        return AVERROR(ENOMEM);
    }
    f->pktq = pktq;
    f->max_size = FFMIN(max_size, FRAME_QUEUE_SIZE);
    f->keep_last = !!keep_last;
    for (i = 0; i < f->max_size; i++)
        if (!(f->queue[i].frame = av_frame_alloc()))
            return AVERROR(ENOMEM);
    return 0;
}

static void frame_queue_destroy(FrameQueue *f)
{
    int i;
    for (i = 0; i < f->max_size; i++) {
        Frame *vp = &f->queue[i];
        frame_queue_unref_item(vp);
        av_frame_free(&vp->frame);
    }
    SDL_DestroyMutex(f->mutex);
    SDL_DestroyCond(f->cond);
}

static void frame_queue_signal(FrameQueue *f)
{
    SDL_LockMutex(f->mutex);
    SDL_CondSignal(f->cond);
    SDL_UnlockMutex(f->mutex);
}

static Frame *frame_queue_peek(FrameQueue *f)
{
    return &f->queue[(f->rindex + f->rindex_shown) % f->max_size];
}

static Frame *frame_queue_peek_next(FrameQueue *f)
{
    return &f->queue[(f->rindex + f->rindex_shown + 1) % f->max_size];
}

static Frame *frame_queue_peek_last(FrameQueue *f)
{
    return &f->queue[f->rindex];
}

static Frame *frame_queue_peek_writable(FrameQueue *f)
{
    /* wait until we have space to put a new frame */
    SDL_LockMutex(f->mutex);
    while (f->size >= f->max_size &&
           !f->pktq->abort_request) {
        SDL_CondWait(f->cond, f->mutex);
    }
    SDL_UnlockMutex(f->mutex);

    if (f->pktq->abort_request)
        return NULL;

    return &f->queue[f->windex];
}

static void frame_queue_push(FrameQueue *f)
{
    if (++f->windex == f->max_size)
        f->windex = 0;
    SDL_LockMutex(f->mutex);
    f->size++;
    SDL_CondSignal(f->cond);
    SDL_UnlockMutex(f->mutex);
}

static void frame_queue_next(FrameQueue *f)
{
    if (f->keep_last && !f->rindex_shown) {
        f->rindex_shown = 1;
        return;
    }
    frame_queue_unref_item(&f->queue[f->rindex]);
    if (++f->rindex == f->max_size)
        f->rindex = 0;
    SDL_LockMutex(f->mutex);
    f->size--;
    SDL_CondSignal(f->cond);
    SDL_UnlockMutex(f->mutex);
}

/* return the number of undisplayed frames in the queue */
static int frame_queue_nb_remaining(FrameQueue *f)
{
    return f->size - f->rindex_shown;
}

/* return last shown position */
static int64_t frame_queue_last_pos(FrameQueue *f)
{
    Frame *fp = &f->queue[f->rindex];
    if (f->rindex_shown && fp->serial == f->pktq->serial)
        return fp->pos;
    else
        return -1;
}

static void decoder_abort(Decoder *d, FrameQueue *fq)
{
    packet_queue_abort(d->queue);
    frame_queue_signal(fq);
    SDL_WaitThread(d->decoder_tid, NULL);
    d->decoder_tid = NULL;
    packet_queue_flush(d->queue);
}

static int realloc_texture(VideoState *is, SDL_Texture **texture, Uint32 new_format, int new_width, int new_height, SDL_BlendMode blendmode, int init_texture)
{
    Uint32 format;
    int access, w, h;
    if (!*texture || SDL_QueryTexture(*texture, &format, &access, &w, &h) < 0 || new_width != w || new_height != h || new_format != format) {
        void *pixels;
        int pitch;
        if (*texture)
            SDL_DestroyTexture(*texture);
        if (!(*texture = SDL_CreateTexture(is->renderer, new_format, SDL_TEXTUREACCESS_STREAMING, new_width, new_height)))
            return -1;
        if (SDL_SetTextureBlendMode(*texture, blendmode) < 0)
            return -1;
        if (init_texture) {
            if (SDL_LockTexture(*texture, NULL, &pixels, &pitch) < 0)
                return -1;
            memset(pixels, 0, pitch * new_height);
            SDL_UnlockTexture(*texture);
        }
        av_log(NULL, AV_LOG_VERBOSE, "Created %dx%d texture with %s.\n", new_width, new_height, SDL_GetPixelFormatName(new_format));
    }
    return 0;
}

static void calculate_display_rect(SDL_Rect *rect,
                                   int scr_xleft, int scr_ytop, int scr_width, int scr_height,
                                   int pic_width, int pic_height, AVRational pic_sar)
{
    AVRational aspect_ratio = pic_sar;
    int64_t width, height, x, y;

    if (av_cmp_q(aspect_ratio, av_make_q(0, 1)) <= 0)
        aspect_ratio = av_make_q(1, 1);

    aspect_ratio = av_mul_q(aspect_ratio, av_make_q(pic_width, pic_height));

    /* XXX: we suppose the screen has a 1.0 pixel ratio */
    height = scr_height;
    width = av_rescale(height, aspect_ratio.num, aspect_ratio.den) & ~1;
    if (width > scr_width) {
        width = scr_width;
        height = av_rescale(width, aspect_ratio.den, aspect_ratio.num) & ~1;
    }
    x = (scr_width - width) / 2;
    y = (scr_height - height) / 2;
    rect->x = scr_xleft + x;
    rect->y = scr_ytop  + y;
    rect->w = FFMAX((int)width,  1);
    rect->h = FFMAX((int)height, 1);
}

static void calculate_content_rect(SDL_Rect *rect,
                                   int scr_xleft, int scr_ytop, double zoom_ratio,
                                   int pic_width, int pic_height)
{
    rect->x = scr_xleft - pic_width * (zoom_ratio - 1) / 2;
    rect->y = scr_ytop - pic_height * (zoom_ratio - 1) / 2;
    rect->w = pic_width * zoom_ratio;
    rect->h = pic_height * zoom_ratio;
}

static void get_sdl_pix_fmt_and_blendmode(int format, Uint32 *sdl_pix_fmt, SDL_BlendMode *sdl_blendmode)
{
    int i;
    *sdl_blendmode = SDL_BLENDMODE_NONE;
    *sdl_pix_fmt = SDL_PIXELFORMAT_UNKNOWN;
    if (format == AV_PIX_FMT_RGB32   ||
        format == AV_PIX_FMT_RGB32_1 ||
        format == AV_PIX_FMT_BGR32   ||
        format == AV_PIX_FMT_BGR32_1)
        *sdl_blendmode = SDL_BLENDMODE_BLEND;
    for (i = 0; i < FF_ARRAY_ELEMS(sdl_texture_format_map) - 1; i++) {
        if (format == sdl_texture_format_map[i].format) {
            *sdl_pix_fmt = sdl_texture_format_map[i].texture_fmt;
            return;
        }
    }
}

static void pack_rgb48le_to_ARGB2101010(AVFrame *frame, uint32_t *dst) {
    uint16_t *src;

    int i, j;
    for (i = 0; i < frame->height; i++) {
        src = (uint16_t *)(frame->data[0] + i * frame->linesize[0]);
        for (j = 0; j < 3 * frame->width; j += 3) {
            const uint32_t r = src[j] >> 6;
            const uint32_t g = src[j + 1] >> 6;
            const uint32_t b = src[j + 2] >> 6;
            *dst = (r << 20) | (g << 10) | (b);
            dst++;
        }
    }
}

static int upload_texture(VideoState *is, SDL_Texture **tex, AVFrame *frame) {
    int ret = 0;
    Uint32 sdl_pix_fmt;
    SDL_BlendMode sdl_blendmode;
    get_sdl_pix_fmt_and_blendmode(frame->format, &sdl_pix_fmt, &sdl_blendmode);
    if (realloc_texture(is, tex, sdl_pix_fmt == SDL_PIXELFORMAT_UNKNOWN ? SDL_PIXELFORMAT_ARGB8888 : sdl_pix_fmt, frame->width, frame->height, sdl_blendmode, 0) < 0)
        return -1;

    switch (sdl_pix_fmt) {
        case SDL_PIXELFORMAT_ARGB2101010:
            pack_rgb48le_to_ARGB2101010(frame, is->texture_buffer);
            ret = SDL_UpdateTexture(*tex, NULL, is->texture_buffer, sizeof(uint32_t) * frame->width);
            break;
        case SDL_PIXELFORMAT_IYUV:
            if (frame->linesize[0] > 0 && frame->linesize[1] > 0 && frame->linesize[2] > 0) {
                ret = SDL_UpdateYUVTexture(*tex, NULL, frame->data[0], frame->linesize[0],
                                                       frame->data[1], frame->linesize[1],
                                                       frame->data[2], frame->linesize[2]);
            } else if (frame->linesize[0] < 0 && frame->linesize[1] < 0 && frame->linesize[2] < 0) {
                ret = SDL_UpdateYUVTexture(*tex, NULL, frame->data[0] + frame->linesize[0] * (frame->height                    - 1), -frame->linesize[0],
                                                       frame->data[1] + frame->linesize[1] * (AV_CEIL_RSHIFT(frame->height, 1) - 1), -frame->linesize[1],
                                                       frame->data[2] + frame->linesize[2] * (AV_CEIL_RSHIFT(frame->height, 1) - 1), -frame->linesize[2]);
            } else {
                av_log(NULL, AV_LOG_ERROR, "Mixed negative and positive linesizes are not supported.\n");
                return -1;
            }
            break;
        default:
            if (frame->linesize[0] < 0) {
                ret = SDL_UpdateTexture(*tex, NULL, frame->data[0] + frame->linesize[0] * (frame->height - 1), -frame->linesize[0]);
            } else {
                ret = SDL_UpdateTexture(*tex, NULL, frame->data[0], frame->linesize[0]);
            }
            break;
    }
    return ret;
}

static enum AVColorSpace sdl_supported_color_spaces[] = {
    AVCOL_SPC_BT709,
    AVCOL_SPC_BT470BG,
    AVCOL_SPC_SMPTE170M,
    AVCOL_SPC_UNSPECIFIED,
};

static void set_sdl_yuv_conversion_mode(AVFrame *frame)
{
#if SDL_VERSION_ATLEAST(2,0,8)
    SDL_YUV_CONVERSION_MODE mode = SDL_YUV_CONVERSION_AUTOMATIC;
    if (frame && (frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUYV422 || frame->format == AV_PIX_FMT_UYVY422)) {
        if (frame->color_range == AVCOL_RANGE_JPEG)
            mode = SDL_YUV_CONVERSION_JPEG;
        else if (frame->colorspace == AVCOL_SPC_BT709)
            mode = SDL_YUV_CONVERSION_BT709;
        else if (frame->colorspace == AVCOL_SPC_BT470BG || frame->colorspace == AVCOL_SPC_SMPTE170M)
            mode = SDL_YUV_CONVERSION_BT601;
    }
    if (frame) {
        av_log(NULL, AV_LOG_VERBOSE, "SDL_YUV_CONVERSION_MODE: %d, for pixel format %s\n", mode, av_get_pix_fmt_name(frame->format));
    }
    SDL_SetYUVConversionMode(mode); /* FIXME: no support for linear transfer */
#endif
}

static void video_image_display(VideoState *is)
{
    Frame *vp;
    SDL_Rect rect;

    vp = frame_queue_peek_last(&is->pictq);

    if (is->window_attached_mode) {
        calculate_display_rect(&rect, is->xleft, is->ytop, is->width, is->height, vp->width, vp->height, vp->sar);
        is->zoom_ratio = FFMIN((double)rect.w / vp->width, (double)rect.h / vp->height);
    } else {
        calculate_content_rect(&rect, is->xleft, is->ytop, is->zoom_ratio, vp->width, vp->height);
    }

    if (is->vk_renderer) {
        double x0, x1, y0, y1;
        x0 = (double)rect.x;
        y0 = (double)rect.y;
        x1 = x0 + rect.w;
        y1 = y0 + rect.h;

        vk_renderer_display_zoom_offset(is->vk_renderer, vp->frame, x0, x1, y0, y1, !no_colorspace_hint);
        return;
    }


    set_sdl_yuv_conversion_mode(vp->frame);
    if (!vp->uploaded) {
        if (upload_texture(is, &is->vid_texture, vp->frame) < 0) {
            set_sdl_yuv_conversion_mode(NULL);
            return;
        }
        vp->uploaded = 1;
        vp->flip_v = vp->frame->linesize[0] < 0;
    }

    SDL_RenderCopyEx(is->renderer, is->vid_texture, NULL, &rect, 0, NULL, vp->flip_v ? SDL_FLIP_VERTICAL : 0);
    set_sdl_yuv_conversion_mode(NULL);
}

static void stream_component_close(VideoState *is, int stream_index)
{
    AVFormatContext *ic = is->ic;
    AVCodecParameters *codecpar;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return;
    codecpar = ic->streams[stream_index]->codecpar;

    switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
        decoder_abort(&is->viddec, &is->pictq);
        decoder_destroy(&is->viddec);
        break;
    default:
        break;
    }

    ic->streams[stream_index]->discard = AVDISCARD_ALL;
    switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
        is->video_st = NULL;
        is->video_stream = -1;
        break;
    default:
        break;
    }
}

static void stream_close(VideoState *is)
{
    /* XXX: use a special url_shutdown call to abort parse cleanly */
    is->abort_request = 1;
    SDL_WaitThread(is->read_tid, NULL);

    /* close each stream */
    if (is->video_stream >= 0)
        stream_component_close(is, is->video_stream);

    avformat_close_input(&is->ic);

    packet_queue_destroy(&is->videoq);

    /* free all pictures */
    frame_queue_destroy(&is->pictq);
    SDL_DestroyCond(is->continue_read_thread);
    av_free(is->filename);
    if (is->vis_texture)
        SDL_DestroyTexture(is->vis_texture);
    if (is->vid_texture)
        SDL_DestroyTexture(is->vid_texture);
    if (is->texture_buffer)
        av_free(is->texture_buffer);
    av_free(is);
}

static void do_exit(void)
{
    VideoState *is;
    int i;

    for (i = 0; i < nb_inputs; i++) {
        is = input_vs[i];
        if (NULL == is) {
            continue;
        }
        if (is->renderer)
            SDL_DestroyRenderer(is->renderer);
        if (is->vk_renderer)
            vk_renderer_destroy(is->vk_renderer);
        if (is->window)
            SDL_DestroyWindow(is->window);
        stream_close(is);
    }
    uninit_opts();
    for (int i = 0; i < nb_vfilters; i++)
        av_freep(&vfilters_list[i]);
    av_freep(&vfilters_list);
    av_freep(&video_codec_name);
    av_freep(&input_filenames);
    avformat_network_deinit();
    if (show_status)
        printf("\n");
    SDL_Quit();
    av_log(NULL, AV_LOG_QUIET, "%s", "");
    exit(0);
}

static void sigterm_handler(int sig)
{
    exit(123);
}

static void set_default_window_size(int width, int height, AVRational sar)
{
    SDL_DisplayMode DM;
    SDL_Rect rect;
    int max_width  = screen_width  ? screen_width  : INT_MAX;
    int max_height = screen_height ? screen_height : INT_MAX;
    if (max_width == INT_MAX && max_height == INT_MAX)
        max_height = height;
    calculate_display_rect(&rect, 0, 0, max_width, max_height, width, height, sar);

    SDL_GetCurrentDisplayMode(0, &DM);
    if (rect.w >= DM.w || rect.h >= DM.h) {
        default_height = DM.h - 135 * DM.h / 1080;
        default_width  = DM.w - 135 * DM.w / 1080;
    } else {
        default_width  = rect.w;
        default_height = rect.h;
    }
}

static int video_open(VideoState *is)
{
    int w,h;

    w = screen_width ? screen_width : default_width;
    h = screen_height ? screen_height : default_height;

    if (master_vs == is) {
        is->window_title = av_asprintf("(master) %s", input_filenames[is->index]);
    } else {
        is->window_title = input_filenames[is->index];
    }

    SDL_SetWindowTitle(is->window, is->window_title);

    SDL_SetWindowSize(is->window, w, h);
    SDL_SetWindowPosition(is->window, screen_left, screen_top);
    if (is_full_screen)
        SDL_SetWindowFullscreen(is->window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_ShowWindow(is->window);

    is->width  = w;
    is->height = h;

    return 0;
}

/* display the current picture, if any */
static void video_display(VideoState *is)
{
    if (!is->width)
        video_open(is);

    SDL_SetRenderDrawColor(is->renderer, 0, 0, 0, 255);
    SDL_RenderClear(is->renderer);
    if (is->video_st)
        video_image_display(is);
    SDL_RenderPresent(is->renderer);
}

static double get_clock(Clock *c)
{
    if (*c->queue_serial != c->serial)
        return NAN;
    if (c->paused) {
        return c->pts;
    } else {
        double time = av_gettime_relative() / 1000000.0;
        return c->pts_drift + time - (time - c->last_updated) * (1.0 - c->speed);
    }
}

static void set_clock_at(Clock *c, double pts, int serial, double time)
{
    c->pts = pts;
    c->last_updated = time;
    c->pts_drift = c->pts - time;
    c->serial = serial;
}

static void set_clock(Clock *c, double pts, int serial)
{
    double time = av_gettime_relative() / 1000000.0;
    set_clock_at(c, pts, serial, time);
}

static void set_clock_speed(Clock *c, double speed)
{
    set_clock(c, get_clock(c), c->serial);
    c->speed = speed;
}

static void init_clock(Clock *c, int *queue_serial)
{
    c->speed = 1.0;
    c->paused = 0;
    c->queue_serial = queue_serial;
    set_clock(c, NAN, -1);
}

static void sync_clock_to_slave(Clock *c, Clock *slave)
{
    double clock = get_clock(c);
    double slave_clock = get_clock(slave);
    if (!isnan(slave_clock) && (isnan(clock) || fabs(clock - slave_clock) > AV_NOSYNC_THRESHOLD))
        set_clock(c, slave_clock, slave->serial);
}

static int get_master_sync_type(VideoState *is) {
    if (is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
        if (is->video_st)
            return AV_SYNC_VIDEO_MASTER;
        else
            return AV_SYNC_EXTERNAL_CLOCK;
    } else {
        return AV_SYNC_EXTERNAL_CLOCK;
    }
}

/* get the current master clock value */
static double get_master_clock(VideoState *is)
{
    double val;

    switch (get_master_sync_type(is)) {
        case AV_SYNC_VIDEO_MASTER:
            val = get_clock(&is->vidclk);
            break;
        default:
            val = get_clock(&is->extclk);
            break;
    }
    return val;
}

static void check_external_clock_speed(VideoState *is) {
   if (is->video_stream >= 0 && is->videoq.nb_packets <= EXTERNAL_CLOCK_MIN_FRAMES) {
       set_clock_speed(&is->extclk, FFMAX(EXTERNAL_CLOCK_SPEED_MIN, is->extclk.speed - EXTERNAL_CLOCK_SPEED_STEP));
   } else if (is->video_stream < 0 || is->videoq.nb_packets > EXTERNAL_CLOCK_MAX_FRAMES) {
       set_clock_speed(&is->extclk, FFMIN(EXTERNAL_CLOCK_SPEED_MAX, is->extclk.speed + EXTERNAL_CLOCK_SPEED_STEP));
   } else {
       double speed = is->extclk.speed;
       if (speed != 1.0)
           set_clock_speed(&is->extclk, speed + EXTERNAL_CLOCK_SPEED_STEP * (1.0 - speed) / fabs(1.0 - speed));
   }
}

/* seek in the stream */
static void stream_seek(VideoState *is, int64_t pos, int64_t rel, int by_bytes)
{
    if (!is->seek_req) {
        is->seek_pos = pos;
        is->seek_rel = rel;
        is->seek_flags &= ~AVSEEK_FLAG_BYTE;
        if (by_bytes)
            is->seek_flags |= AVSEEK_FLAG_BYTE;
        is->seek_req = 1;
        SDL_CondSignal(is->continue_read_thread);
    }
}

/* pause or resume the video */
static void stream_pause(VideoState *is, int pause)
{
    if (is->paused) {
        is->frame_timer += av_gettime_relative() / 1000000.0 - is->vidclk.last_updated;
        if (is->read_pause_return != AVERROR(ENOSYS)) {
            is->vidclk.paused = 0;
        }
        set_clock(&is->vidclk, get_clock(&is->vidclk), is->vidclk.serial);
    }
    set_clock(&is->extclk, get_clock(&is->extclk), is->extclk.serial);
    is->paused = is->vidclk.paused = is->extclk.paused = pause;
}

static void toggle_pause(VideoState *is, int all)
{
    int i;
    int pause = !is->paused;
    if (!all) {
        stream_pause(is, pause);
        is->step = 0;
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        stream_pause(input_vs[i], pause);
        input_vs[i]->step = 0;
    }
}

static void step_to_next_frame(VideoState *is)
{
    int i;
    if (is) {
        /* if the stream is paused unpause it, then step */
        if (is->paused)
            stream_pause(is, !is->paused);
        is->step = 1;
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        is = input_vs[i];
        if (is->paused)
            stream_pause(is, !is->paused);
        is->step = 1;
    }
}

static double compute_target_delay(double delay, VideoState *is)
{
    double sync_threshold, diff = 0;

    /* update delay to follow master synchronisation source */
    if (get_master_sync_type(is) != AV_SYNC_VIDEO_MASTER) {
        /* if video is slave, we try to correct big delays by
           duplicating or deleting a frame */
        diff = get_clock(&is->vidclk) - get_master_clock(is);

        /* skip or repeat frame. We take into account the
           delay to compute the threshold. I still don't know
           if it is the best guess */
        sync_threshold = FFMAX(AV_SYNC_THRESHOLD_MIN, FFMIN(AV_SYNC_THRESHOLD_MAX, delay));
        if (!isnan(diff) && fabs(diff) < is->max_frame_duration) {
            if (diff <= -sync_threshold)
                delay = FFMAX(0, delay + diff);
            else if (diff >= sync_threshold && delay > AV_SYNC_FRAMEDUP_THRESHOLD)
                delay = delay + diff;
            else if (diff >= sync_threshold)
                delay = 2 * delay;
        }
    }

    av_log(NULL, AV_LOG_TRACE, "video: delay=%0.3f A-V=%f\n",
            delay, -diff);

    return delay;
}

static double vp_duration(VideoState *is, Frame *vp, Frame *nextvp) {
    if (vp->serial == nextvp->serial) {
        double duration = nextvp->pts - vp->pts;
        if (isnan(duration) || duration <= 0 || duration > is->max_frame_duration)
            return vp->duration;
        else
            return duration;
    } else {
        return 0.0;
    }
}

static void update_video_pts(VideoState *is, double pts, int serial)
{
    /* update current video pts */
    set_clock(&is->vidclk, pts, serial);
    sync_clock_to_slave(&is->extclk, &is->vidclk);
}

/* called to display each frame */
static void video_refresh(double *remaining_time)
{
    VideoState *is;
    int i;
    double time;
    double last_duration, duration, delay;
    Frame *vp, *lastvp;

    if (!master_vs->paused && get_master_sync_type(master_vs) == AV_SYNC_EXTERNAL_CLOCK && master_vs->realtime)
        check_external_clock_speed(master_vs);

    if (master_vs->video_st) {
retry:
        for (i = 0; i < nb_inputs; i++) {
            is = input_vs[i];
            if (frame_queue_nb_remaining(&is->pictq) == 0) {
                // if not all streams have got pictures
                goto pre_display;
            }
        }
        // all streams got picture ready, playing in sync with master stream

        /* dequeue the picture */
        lastvp = frame_queue_peek_last(&master_vs->pictq);
        vp = frame_queue_peek(&master_vs->pictq);

        if (vp->serial != master_vs->videoq.serial) {
            for (i = 0; i < nb_inputs; i++) {
                is = input_vs[i];
                frame_queue_next(&is->pictq);
            }
            goto retry;
        }

        if (lastvp->serial != vp->serial)
            master_vs->frame_timer = av_gettime_relative() / 1000000.0;

        // master is paused, cannot playing in sync
        if (master_vs->paused)
            goto pre_display;

        /* compute nominal last_duration */
        last_duration = vp_duration(master_vs, lastvp, vp);
        delay = compute_target_delay(last_duration, master_vs);

        time = av_gettime_relative()/1000000.0;
        if (time < master_vs->frame_timer + delay) {
            *remaining_time = FFMIN(master_vs->frame_timer + delay - time, *remaining_time);
            goto display;
        }

        master_vs->frame_timer += delay;
        if (delay > 0 && time - master_vs->frame_timer > AV_SYNC_THRESHOLD_MAX)
            master_vs->frame_timer = time;

        SDL_LockMutex(master_vs->pictq.mutex);
        if (!isnan(vp->pts))
            update_video_pts(master_vs, vp->pts, vp->serial);
        SDL_UnlockMutex(master_vs->pictq.mutex);

        if (frame_queue_nb_remaining(&master_vs->pictq) > 1) {
            Frame *nextvp = frame_queue_peek_next(&master_vs->pictq);
            duration = vp_duration(master_vs, vp, nextvp);
            if(!master_vs->step &&
                    (framedrop > 0 || (framedrop && get_master_sync_type(master_vs) != AV_SYNC_VIDEO_MASTER)) &&
                    time > master_vs->frame_timer + duration){
                for (i = 0; i < nb_inputs; i++) {
                    is = input_vs[i];
                    is->frame_drops_late++;
                    frame_queue_next(&is->pictq);
                }
                goto retry;
            }
        }

pre_display:
        for (i = 0; i < nb_inputs; i++) {
            is = input_vs[i];
            // only the stream, which is not paused and ready, could play
            if (!is->paused && frame_queue_nb_remaining(&is->pictq) > 0) {
                Frame *vp;
                vp = frame_queue_peek_last(&is->pictq);
                av_log(NULL, AV_LOG_TRACE,
                       "Window %d: displayed %d frames, frame_type=%c frame_pts=%"PRId64", pts=%0.3f\n",
                       is->index, is->frame_displayed, av_get_picture_type_char(vp->frame->pict_type),
                       vp->frame->pts, vp->pts);
                is->frame_displayed++;

                frame_queue_next(&is->pictq);
                is->force_refresh = 1;
                if (is->step)
                    stream_pause(is, !is->paused);
            }
        }

display:
        /* display picture */
        for (i = 0; i < nb_inputs; i++) {
            is = input_vs[i];
            if (is->force_refresh && is->pictq.rindex_shown)
                video_display(is);
        }
    }
    for (i = 0; i < nb_inputs; i++) {
        is = input_vs[i];
        is->force_refresh = 0;
    }

    if (show_status) {
        AVBPrint buf;
        static int64_t last_time;
        int64_t cur_time;
        int vqsize;

        cur_time = av_gettime_relative();
        if (!last_time || (cur_time - last_time) >= 30000) {
            vqsize = 0;
            if (master_vs->video_st)
                vqsize = master_vs->videoq.size;

            av_bprint_init(&buf, 0, AV_BPRINT_SIZE_AUTOMATIC);
            av_bprintf(&buf,
                   "%7.2f fd=%4d vq=%5dKB no=%d zoom=%.2f%%    \r",
                   get_master_clock(master_vs),
                   master_vs->frame_drops_early + master_vs->frame_drops_late,
                   vqsize / 1024,
                   master_vs->frame_displayed,
                   master_vs->zoom_ratio * 100.0);

            if (show_status == 1 && AV_LOG_INFO > av_log_get_level())
                fprintf(stderr, "%s", buf.str);
            else
                av_log(NULL, AV_LOG_INFO, "%s", buf.str);

            fflush(stderr);
            av_bprint_finalize(&buf, NULL);
            last_time = cur_time;
        }
    }
}

static int queue_picture(VideoState *is, AVFrame *src_frame, double pts, double duration, int64_t pos, int serial)
{
    Frame *vp;

    av_log(NULL, AV_LOG_DEBUG,
           "Window %d: queued %d frames, frame_type=%c frame_pts=%"PRId64", pts=%0.3f\n",
           is->index, is->frame_queued, av_get_picture_type_char(src_frame->pict_type),
           src_frame->pts, pts);

    if (!(vp = frame_queue_peek_writable(&is->pictq)))
        return -1;

    vp->sar = src_frame->sample_aspect_ratio;
    vp->uploaded = 0;

    vp->width = src_frame->width;
    vp->height = src_frame->height;
    vp->format = src_frame->format;

    vp->pts = pts;
    vp->duration = duration;
    vp->pos = pos;
    vp->serial = serial;

    set_default_window_size(vp->width, vp->height, vp->sar);

    av_frame_move_ref(vp->frame, src_frame);
    frame_queue_push(&is->pictq);
    is->frame_queued++;
    return 0;
}

static int get_video_frame(VideoState *is, AVFrame *frame)
{
    int got_picture;

    if ((got_picture = decoder_decode_frame(&is->viddec, frame)) < 0)
        return -1;

    if (got_picture) {
        double dpts = NAN;

        if (frame->pts != AV_NOPTS_VALUE)
            dpts = av_q2d(is->video_st->time_base) * frame->pts;

        frame->sample_aspect_ratio = av_guess_sample_aspect_ratio(is->ic, is->video_st, frame);

        if (framedrop>0 || (framedrop && get_master_sync_type(is) != AV_SYNC_VIDEO_MASTER)) {
            if (frame->pts != AV_NOPTS_VALUE) {
                double diff = dpts - get_master_clock(is);
                if (!isnan(diff) && fabs(diff) < AV_NOSYNC_THRESHOLD &&
                    diff - is->frame_last_filter_delay < 0 &&
                    is->viddec.pkt_serial == is->vidclk.serial &&
                    is->videoq.nb_packets) {
                    is->frame_drops_early++;
                    av_frame_unref(frame);
                    got_picture = 0;
                }
            }
        }
    }

    return got_picture;
}

static int configure_filtergraph(AVFilterGraph *graph, const char *filtergraph,
                                 AVFilterContext *source_ctx, AVFilterContext *sink_ctx)
{
    int ret, i;
    int nb_filters = graph->nb_filters;
    AVFilterInOut *outputs = NULL, *inputs = NULL;

    if (filtergraph) {
        outputs = avfilter_inout_alloc();
        inputs  = avfilter_inout_alloc();
        if (!outputs || !inputs) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        outputs->name       = av_strdup("in");
        outputs->filter_ctx = source_ctx;
        outputs->pad_idx    = 0;
        outputs->next       = NULL;

        inputs->name        = av_strdup("out");
        inputs->filter_ctx  = sink_ctx;
        inputs->pad_idx     = 0;
        inputs->next        = NULL;

        if ((ret = avfilter_graph_parse_ptr(graph, filtergraph, &inputs, &outputs, NULL)) < 0)
            goto fail;
    } else {
        if ((ret = avfilter_link(source_ctx, 0, sink_ctx, 0)) < 0)
            goto fail;
    }

    /* Reorder the filters to ensure that inputs of the custom filters are merged first */
    for (i = 0; i < graph->nb_filters - nb_filters; i++)
        FFSWAP(AVFilterContext*, graph->filters[i], graph->filters[i + nb_filters]);

    ret = avfilter_graph_config(graph, NULL);
fail:
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    return ret;
}

static int need_10bit_rendering(const VideoState *is, const AVFrame *frame)
{
    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(frame->format);
    int c;
    int frame_need_10bit = 0;

    for (c = 0; c < pixdesc->nb_components; c++) {
        if (pixdesc->comp[c].depth > 8) {
            frame_need_10bit = 1;
            break;
        }
    }

    if (0 == use_10bit) {
        if (frame_need_10bit && !is->vk_renderer) {
            av_log(NULL, AV_LOG_WARNING, "%s: rendering %s in 8bit may affect image quality.\n",
                   is->filename, pixdesc->name);
        }
        return 0;
    }

    if (0 < use_10bit) {
        if (!frame_need_10bit) {
            av_log(NULL, AV_LOG_WARNING, "%s: rendering %s in 10bit may affect performance.\n",
                   is->filename, pixdesc->name);
        }
        return 1;
    }

    // use_10bit < -1: auto
    if (frame_need_10bit) {
        if (is->vk_renderer) {
            av_log(NULL, AV_LOG_VERBOSE, "%s: leave %s rendering to Vulkan backend.\n",
                   is->filename, pixdesc->name);
            return 0;
        }
        av_log(NULL, AV_LOG_VERBOSE, "%s: SDL use 10bit (rgb48) to render %s.\n",
               is->filename, pixdesc->name);
    }
    return frame_need_10bit;
}

static int configure_video_filters(AVFilterGraph *graph, VideoState *is, const char *vfilters, AVFrame *frame)
{
    enum AVPixelFormat pix_fmts[FF_ARRAY_ELEMS(sdl_texture_format_map)];
    const AVPixFmtDescriptor *pixdesc;
    char sws_flags_str[512] = "flags=spline+full_chroma_int+accurate_rnd+full_chroma_inp:";
    char buffersrc_args[256];
    int ret;
    AVFilterContext *filt_src = NULL, *filt_out = NULL, *last_filter = NULL;
    AVCodecParameters *codecpar = is->video_st->codecpar;
    AVRational fr = av_guess_frame_rate(is->ic, is->video_st, NULL);
    const AVDictionaryEntry *e = NULL;
    int nb_pix_fmts = 0;
    int i, j;
    AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();

    if (!par)
        return AVERROR(ENOMEM);

    if (need_10bit_rendering(is, frame)) {
        pix_fmts[nb_pix_fmts++] = AV_PIX_FMT_RGB48LE;
        if (is->texture_buffer)
            av_free(is->texture_buffer);
        // prepare texture buffer for 10bit
        is->texture_buffer = av_malloc(sizeof(uint32_t) * frame->width * frame->height);
    } else {
        for (i = 0; i < is->renderer_info.num_texture_formats; i++) {
            for (j = 0; j < FF_ARRAY_ELEMS(sdl_texture_format_map) - 1; j++) {
                if (is->renderer_info.texture_formats[i] != sdl_texture_format_map[j].texture_fmt) {
                    continue;
                }

                pixdesc = av_pix_fmt_desc_get(sdl_texture_format_map[j].format);
                if (AVCOL_RANGE_MPEG == frame->color_range && !(pixdesc->flags & AV_PIX_FMT_FLAG_RGB)) {
                    // SDL does not process limited range, so we need to convert it by ourselves.
                    continue;
                }

                pix_fmts[nb_pix_fmts++] = sdl_texture_format_map[j].format;
                break;
            }
        }
    }
    pix_fmts[nb_pix_fmts] = AV_PIX_FMT_NONE;

    while ((e = av_dict_iterate(sws_dict, e))) {
        if (!strcmp(e->key, "sws_flags")) {
            av_strlcatf(sws_flags_str, sizeof(sws_flags_str), "%s=%s:", "flags", e->value);
        } else
            av_strlcatf(sws_flags_str, sizeof(sws_flags_str), "%s=%s:", e->key, e->value);
    }
    if (strlen(sws_flags_str))
        sws_flags_str[strlen(sws_flags_str)-1] = '\0';

    graph->scale_sws_opts = av_strdup(sws_flags_str);

    snprintf(buffersrc_args, sizeof(buffersrc_args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d:"
             "colorspace=%d:range=%d",
             frame->width, frame->height, frame->format,
             is->video_st->time_base.num, is->video_st->time_base.den,
             codecpar->sample_aspect_ratio.num, FFMAX(codecpar->sample_aspect_ratio.den, 1),
             frame->colorspace, frame->color_range);
    if (fr.num && fr.den)
        av_strlcatf(buffersrc_args, sizeof(buffersrc_args), ":frame_rate=%d/%d", fr.num, fr.den);

    if ((ret = avfilter_graph_create_filter(&filt_src,
                                            avfilter_get_by_name("buffer"),
                                            "ffcmpr_buffer", buffersrc_args, NULL,
                                            graph)) < 0)
        goto fail;
    par->hw_frames_ctx = frame->hw_frames_ctx;
    ret = av_buffersrc_parameters_set(filt_src, par);
    if (ret < 0)
        goto fail;

    ret = avfilter_graph_create_filter(&filt_out,
                                       avfilter_get_by_name("buffersink"),
                                       "ffcmpr_buffersink", NULL, NULL, graph);
    if (ret < 0)
        goto fail;

    if ((ret = av_opt_set_int_list(filt_out, "pix_fmts", pix_fmts,  AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN)) < 0)
        goto fail;
    if (!is->vk_renderer &&
        (ret = av_opt_set_int_list(filt_out, "color_spaces", sdl_supported_color_spaces,  AVCOL_SPC_UNSPECIFIED, AV_OPT_SEARCH_CHILDREN)) < 0)
        goto fail;

    last_filter = filt_out;

/* Note: this macro adds a filter before the lastly added filter, so the
 * processing order of the filters is in reverse */
#define INSERT_FILT(name, arg) do {                                          \
    AVFilterContext *filt_ctx;                                               \
                                                                             \
    ret = avfilter_graph_create_filter(&filt_ctx,                            \
                                       avfilter_get_by_name(name),           \
                                       "ffcmpr_" name, arg, NULL, graph);    \
    if (ret < 0)                                                             \
        goto fail;                                                           \
                                                                             \
    ret = avfilter_link(filt_ctx, 0, last_filter, 0);                        \
    if (ret < 0)                                                             \
        goto fail;                                                           \
                                                                             \
    last_filter = filt_ctx;                                                  \
} while (0)

    if (autorotate) {
        double theta = 0.0;
        int32_t *displaymatrix = NULL;
        AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DISPLAYMATRIX);
        if (sd)
            displaymatrix = (int32_t *)sd->data;
        if (!displaymatrix) {
            const AVPacketSideData *psd = av_packet_side_data_get(is->video_st->codecpar->coded_side_data,
                                                                  is->video_st->codecpar->nb_coded_side_data,
                                                                  AV_PKT_DATA_DISPLAYMATRIX);
            if (psd)
                displaymatrix = (int32_t *)psd->data;
        }
        theta = get_rotation(displaymatrix);

        if (fabs(theta - 90) < 1.0) {
            INSERT_FILT("transpose", displaymatrix[3] > 0 ? "cclock_flip" : "clock");
        } else if (fabs(theta - 180) < 1.0) {
            if (displaymatrix[0] < 0)
                INSERT_FILT("hflip", NULL);
            if (displaymatrix[4] < 0)
                INSERT_FILT("vflip", NULL);
        } else if (fabs(theta - 270) < 1.0) {
            INSERT_FILT("transpose", displaymatrix[3] < 0 ? "clock_flip" : "cclock");
        } else if (fabs(theta) > 1.0) {
            char rotate_buf[64];
            snprintf(rotate_buf, sizeof(rotate_buf), "%f*PI/180", theta);
            INSERT_FILT("rotate", rotate_buf);
        } else {
            if (displaymatrix && displaymatrix[4] < 0)
                INSERT_FILT("vflip", NULL);
        }
    }

    if ((ret = configure_filtergraph(graph, vfilters, filt_src, last_filter)) < 0)
        goto fail;

    is->in_video_filter  = filt_src;
    is->out_video_filter = filt_out;

fail:
    av_freep(&par);
    return ret;
}

static int decoder_start(Decoder *d, int (*fn)(void *), const char *thread_name, void* arg)
{
    packet_queue_start(d->queue);
    d->decoder_tid = SDL_CreateThread(fn, thread_name, arg);
    if (!d->decoder_tid) {
        av_log(NULL, AV_LOG_ERROR, "SDL_CreateThread(): %s\n", SDL_GetError());
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int video_thread(void *arg)
{
    VideoState *is = arg;
    AVFrame *frame = av_frame_alloc();
    double pts;
    double duration;
    int ret;
    AVRational tb = is->video_st->time_base;
    AVRational frame_rate = av_guess_frame_rate(is->ic, is->video_st, NULL);

    AVFilterGraph *graph = NULL;
    AVFilterContext *filt_out = NULL, *filt_in = NULL;
    int last_w = 0;
    int last_h = 0;
    enum AVPixelFormat last_format = -2;
    int last_serial = -1;
    int last_vfilter_idx = 0;

    if (!frame)
        return AVERROR(ENOMEM);

    for (;;) {
        ret = get_video_frame(is, frame);
        if (ret < 0)
            goto the_end;

        if (!ret) {
            av_log(NULL, AV_LOG_WARNING,
                   "Window %d (%s) droped a bad frame, frame sync may be broken!\n",
                   is->index, is->window_title);
            continue;
        }

        if (   last_w != frame->width
            || last_h != frame->height
            || last_format != frame->format
            || last_serial != is->viddec.pkt_serial
            || last_vfilter_idx != is->vfilter_idx) {
            av_log(NULL, AV_LOG_DEBUG,
                   "Video frame changed from size:%dx%d format:%s serial:%d to size:%dx%d format:%s serial:%d\n",
                   last_w, last_h,
                   (const char *)av_x_if_null(av_get_pix_fmt_name(last_format), "none"), last_serial,
                   frame->width, frame->height,
                   (const char *)av_x_if_null(av_get_pix_fmt_name(frame->format), "none"), is->viddec.pkt_serial);
            avfilter_graph_free(&graph);
            graph = avfilter_graph_alloc();
            if (!graph) {
                ret = AVERROR(ENOMEM);
                goto the_end;
            }
            graph->nb_threads = filter_nbthreads;
            if ((ret = configure_video_filters(graph, is, vfilters_list ? vfilters_list[is->vfilter_idx] : NULL, frame)) < 0) {
                SDL_Event event;
                event.type = FF_QUIT_EVENT;
                event.user.data1 = is;
                SDL_PushEvent(&event);
                goto the_end;
            }
            filt_in  = is->in_video_filter;
            filt_out = is->out_video_filter;
            last_w = frame->width;
            last_h = frame->height;
            last_format = frame->format;
            last_serial = is->viddec.pkt_serial;
            last_vfilter_idx = is->vfilter_idx;
            frame_rate = av_buffersink_get_frame_rate(filt_out);
        }

        ret = av_buffersrc_add_frame(filt_in, frame);
        if (ret < 0)
            goto the_end;

        while (ret >= 0) {
            FrameData *fd;

            is->frame_last_returned_time = av_gettime_relative() / 1000000.0;

            ret = av_buffersink_get_frame_flags(filt_out, frame, 0);
            if (ret < 0) {
                if (ret == AVERROR_EOF)
                    is->viddec.finished = is->viddec.pkt_serial;
                ret = 0;
                break;
            }

            fd = frame->opaque_ref ? (FrameData*)frame->opaque_ref->data : NULL;

            is->frame_last_filter_delay = av_gettime_relative() / 1000000.0 - is->frame_last_returned_time;
            if (fabs(is->frame_last_filter_delay) > AV_NOSYNC_THRESHOLD / 10.0)
                is->frame_last_filter_delay = 0;
            tb = av_buffersink_get_time_base(filt_out);
            duration = (frame_rate.num && frame_rate.den ? av_q2d((AVRational){frame_rate.den, frame_rate.num}) : 0);
            pts = (frame->pts == AV_NOPTS_VALUE) ? NAN : frame->pts * av_q2d(tb);
            ret = queue_picture(is, frame, pts, duration, fd ? fd->pkt_pos : -1, is->viddec.pkt_serial);
            av_frame_unref(frame);
            if (is->videoq.serial != is->viddec.pkt_serial)
                break;
        }

        if (ret < 0)
            goto the_end;
    }
 the_end:
    avfilter_graph_free(&graph);
    av_frame_free(&frame);
    return 0;
}

static int create_hwaccel(AVBufferRef **device_ctx, VkRenderer *vk_renderer)
{
    enum AVHWDeviceType type;
    int ret;
    AVBufferRef *vk_dev;

    *device_ctx = NULL;

    if (!hwaccel)
        return 0;

    type = av_hwdevice_find_type_by_name(hwaccel);
    if (type == AV_HWDEVICE_TYPE_NONE)
        return AVERROR(ENOTSUP);

    ret = vk_renderer_get_hw_dev(vk_renderer, &vk_dev);
    if (ret < 0)
        return ret;

    ret = av_hwdevice_ctx_create_derived(device_ctx, type, vk_dev, 0);
    if (!ret)
        return 0;

    if (ret != AVERROR(ENOSYS))
        return ret;

    av_log(NULL, AV_LOG_WARNING, "Derive %s from vulkan not supported.\n", hwaccel);
    ret = av_hwdevice_ctx_create(device_ctx, type, NULL, NULL, 0);
    return ret;
}

/* open a given stream. Return 0 if OK */
static int stream_component_open(VideoState *is, int stream_index)
{
    AVFormatContext *ic = is->ic;
    AVCodecContext *avctx;
    const AVCodec *codec;
    const char *forced_codec_name = NULL;
    AVDictionary *opts = NULL;
    int ret = 0;
    int stream_lowres = lowres;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return -1;

    avctx = avcodec_alloc_context3(NULL);
    if (!avctx)
        return AVERROR(ENOMEM);

    ret = avcodec_parameters_to_context(avctx, ic->streams[stream_index]->codecpar);
    if (ret < 0)
        goto fail;
    avctx->pkt_timebase = ic->streams[stream_index]->time_base;

    codec = avcodec_find_decoder(avctx->codec_id);

    switch(avctx->codec_type){
        case AVMEDIA_TYPE_VIDEO   : is->last_video_stream    = stream_index; forced_codec_name =    video_codec_name; break;
    }
    if (forced_codec_name)
        codec = avcodec_find_decoder_by_name(forced_codec_name);
    if (!codec) {
        if (forced_codec_name) av_log(NULL, AV_LOG_WARNING,
                                      "No codec could be found with name '%s'\n", forced_codec_name);
        else                   av_log(NULL, AV_LOG_WARNING,
                                      "No decoder could be found for codec %s\n", avcodec_get_name(avctx->codec_id));
        ret = AVERROR(EINVAL);
        goto fail;
    }

    avctx->codec_id = codec->id;
    if (stream_lowres > codec->max_lowres) {
        av_log(avctx, AV_LOG_WARNING, "The maximum value for lowres supported by the decoder is %d\n",
                codec->max_lowres);
        stream_lowres = codec->max_lowres;
    }
    avctx->lowres = stream_lowres;

    if (fast)
        avctx->flags2 |= AV_CODEC_FLAG2_FAST;

    ret = filter_codec_opts(codec_opts, avctx->codec_id, ic,
                            ic->streams[stream_index], codec, &opts, NULL);
    if (ret < 0)
        goto fail;

    if (!av_dict_get(opts, "threads", NULL, 0))
        av_dict_set(&opts, "threads", "auto", 0);
    if (stream_lowres)
        av_dict_set_int(&opts, "lowres", stream_lowres, 0);

    av_dict_set(&opts, "flags", "+copy_opaque", AV_DICT_MULTIKEY);

    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) {
        ret = create_hwaccel(&avctx->hw_device_ctx, is->vk_renderer);
        if (ret < 0)
            goto fail;
    }

    if ((ret = avcodec_open2(avctx, codec, &opts)) < 0) {
        goto fail;
    }
    ret = check_avoptions(opts);
    if (ret < 0)
        goto fail;

    is->eof = 0;
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    switch (avctx->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
        is->video_stream = stream_index;
        is->video_st = ic->streams[stream_index];

        if ((ret = decoder_init(&is->viddec, avctx, &is->videoq, is->continue_read_thread)) < 0)
            goto fail;
        if ((ret = decoder_start(&is->viddec, video_thread, "video_decoder", is)) < 0)
            goto out;
        is->queue_attachments_req = 1;
        break;
    default:
        break;
    }
    goto out;

fail:
    avcodec_free_context(&avctx);
out:
    av_dict_free(&opts);

    return ret;
}

static int decode_interrupt_cb(void *ctx)
{
    VideoState *is = ctx;
    return is->abort_request;
}

static int stream_has_enough_packets(AVStream *st, int stream_id, PacketQueue *queue) {
    return stream_id < 0 ||
           queue->abort_request ||
           (st->disposition & AV_DISPOSITION_ATTACHED_PIC) ||
           queue->nb_packets > MIN_FRAMES && (!queue->duration || av_q2d(st->time_base) * queue->duration > 1.0);
}

static int is_realtime(AVFormatContext *s)
{
    if(   !strcmp(s->iformat->name, "rtp")
       || !strcmp(s->iformat->name, "rtsp")
       || !strcmp(s->iformat->name, "sdp")
    )
        return 1;

    if(s->pb && (   !strncmp(s->url, "rtp:", 4)
                 || !strncmp(s->url, "udp:", 4)
                )
    )
        return 1;
    return 0;
}

/* this thread gets the stream from the disk or the network */
static int read_thread(void *arg)
{
    VideoState *is = arg;
    AVFormatContext *ic = NULL;
    int err, i, ret;
    int st_index[AVMEDIA_TYPE_NB];
    AVPacket *pkt = NULL;
    SDL_mutex *wait_mutex = SDL_CreateMutex();
    int scan_all_pmts_set = 0;

    if (!wait_mutex) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    memset(st_index, -1, sizeof(st_index));
    is->last_video_stream = is->video_stream = -1;
    is->eof = 0;

    pkt = av_packet_alloc();
    if (!pkt) {
        av_log(NULL, AV_LOG_FATAL, "Could not allocate packet.\n");
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ic = avformat_alloc_context();
    if (!ic) {
        av_log(NULL, AV_LOG_FATAL, "Could not allocate context.\n");
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ic->interrupt_callback.callback = decode_interrupt_cb;
    ic->interrupt_callback.opaque = is;
    if (!av_dict_get(is->format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE)) {
        av_dict_set(&is->format_opts, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);
        scan_all_pmts_set = 1;
    }
    err = avformat_open_input(&ic, is->filename, is->iformat, &is->format_opts);
    if (err < 0) {
        print_error(is->filename, err);
        ret = -1;
        goto fail;
    }
    if (scan_all_pmts_set)
        av_dict_set(&is->format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE);
    remove_avoptions(&is->format_opts, codec_opts);

    ret = check_avoptions(is->format_opts);
    if (ret < 0)
        goto fail;
    is->ic = ic;

    if (genpts)
        ic->flags |= AVFMT_FLAG_GENPTS;

    if (find_stream_info) {
        AVDictionary **opts;
        int orig_nb_streams = ic->nb_streams;

        err = setup_find_stream_info_opts(ic, codec_opts, &opts);
        if (err < 0) {
            av_log(NULL, AV_LOG_ERROR,
                   "Error setting up avformat_find_stream_info() options\n");
            ret = err;
            goto fail;
        }

        err = avformat_find_stream_info(ic, opts);

        for (i = 0; i < orig_nb_streams; i++)
            av_dict_free(&opts[i]);
        av_freep(&opts);

        if (err < 0) {
            av_log(NULL, AV_LOG_WARNING,
                   "%s: could not find codec parameters\n", is->filename);
            ret = -1;
            goto fail;
        }
    }

    if (ic->pb)
        ic->pb->eof_reached = 0; // FIXME hack, ffplay maybe should not use avio_feof() to test for the end

    if (seek_by_bytes < 0)
        seek_by_bytes = !(ic->iformat->flags & AVFMT_NO_BYTE_SEEK) &&
                        !!(ic->iformat->flags & AVFMT_TS_DISCONT) &&
                        strcmp("ogg", ic->iformat->name);

    is->max_frame_duration = (ic->iformat->flags & AVFMT_TS_DISCONT) ? 10.0 : 3600.0;

    /* if seeking requested, we execute it */
    if (start_time != AV_NOPTS_VALUE) {
        int64_t timestamp;

        timestamp = start_time;
        /* add the stream start time */
        if (ic->start_time != AV_NOPTS_VALUE)
            timestamp += ic->start_time;
        ret = avformat_seek_file(ic, -1, INT64_MIN, timestamp, INT64_MAX, 0);
        if (ret < 0) {
            av_log(NULL, AV_LOG_WARNING, "%s: could not seek to position %0.3f\n",
                    is->filename, (double)timestamp / AV_TIME_BASE);
        }
    }

    is->realtime = is_realtime(ic);

    if (show_status)
        av_dump_format(ic, 0, is->filename, 0);

    for (i = 0; i < ic->nb_streams; i++) {
        AVStream *st = ic->streams[i];
        enum AVMediaType type = st->codecpar->codec_type;
        st->discard = AVDISCARD_ALL;
        if (type >= 0 && wanted_stream_spec[type] && st_index[type] == -1)
            if (avformat_match_stream_specifier(ic, st, wanted_stream_spec[type]) > 0)
                st_index[type] = i;
    }
    for (i = 0; i < AVMEDIA_TYPE_NB; i++) {
        if (wanted_stream_spec[i] && st_index[i] == -1) {
            av_log(NULL, AV_LOG_ERROR, "Stream specifier %s does not match any %s stream\n", wanted_stream_spec[i], av_get_media_type_string(i));
            st_index[i] = INT_MAX;
        }
    }

    st_index[AVMEDIA_TYPE_VIDEO] =
        av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO,
                            st_index[AVMEDIA_TYPE_VIDEO], -1, NULL, 0);
    if (st_index[AVMEDIA_TYPE_VIDEO] >= 0) {
        AVStream *st = ic->streams[st_index[AVMEDIA_TYPE_VIDEO]];
        AVCodecParameters *codecpar = st->codecpar;
        AVRational sar = av_guess_sample_aspect_ratio(ic, st, NULL);
        if (codecpar->width)
            set_default_window_size(codecpar->width, codecpar->height, sar);
    }

    ret = -1;
    if (st_index[AVMEDIA_TYPE_VIDEO] >= 0) {
        ret = stream_component_open(is, st_index[AVMEDIA_TYPE_VIDEO]);
    }

    if (is->video_stream < 0) {
        av_log(NULL, AV_LOG_FATAL, "Failed to open file '%s' or configure filtergraph\n",
               is->filename);
        ret = -1;
        goto fail;
    }

    if (infinite_buffer < 0 && is->realtime)
        infinite_buffer = 1;

    for (;;) {
        if (is->abort_request)
            break;
        if (is->paused != is->last_paused) {
            is->last_paused = is->paused;
            if (is->paused)
                is->read_pause_return = av_read_pause(ic);
            else
                av_read_play(ic);
        }
#if CONFIG_RTSP_DEMUXER || CONFIG_MMSH_PROTOCOL
        if (is->paused &&
                (!strcmp(ic->iformat->name, "rtsp") ||
                 (ic->pb && !strncmp(input_filenames[is->index], "mmsh:", 5)))) {
            /* wait 10 ms to avoid trying to get another packet */
            /* XXX: horrible */
            SDL_Delay(10);
            continue;
        }
#endif
        if (is->seek_req) {
            int64_t seek_target = is->seek_pos;
            int64_t seek_min    = is->seek_rel > 0 ? seek_target - is->seek_rel + 2: INT64_MIN;
            int64_t seek_max    = is->seek_rel < 0 ? seek_target - is->seek_rel - 2: INT64_MAX;
// FIXME the +-2 is due to rounding being not done in the correct direction in generation
//      of the seek_pos/seek_rel variables

            ret = avformat_seek_file(is->ic, -1, seek_min, seek_target, seek_max, is->seek_flags);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR,
                       "%s: error while seeking\n", is->ic->url);
            } else {
                if (is->video_stream >= 0)
                    packet_queue_flush(&is->videoq);
                if (is->seek_flags & AVSEEK_FLAG_BYTE) {
                   set_clock(&is->extclk, NAN, 0);
                } else {
                   set_clock(&is->extclk, seek_target / (double)AV_TIME_BASE, 0);
                }
            }
            is->seek_req = 0;
            is->queue_attachments_req = 1;
            is->eof = 0;
            if (is->paused)
                step_to_next_frame(is);
        }
        if (is->queue_attachments_req) {
            if (is->video_st && is->video_st->disposition & AV_DISPOSITION_ATTACHED_PIC) {
                if ((ret = av_packet_ref(pkt, &is->video_st->attached_pic)) < 0)
                    goto fail;
                packet_queue_put(&is->videoq, pkt);
                packet_queue_put_nullpacket(&is->videoq, pkt, is->video_stream);
            }
            is->queue_attachments_req = 0;
        }

        /* if the queue are full, no need to read more */
        if (infinite_buffer<1 &&
              (is->videoq.size > MAX_QUEUE_SIZE
            || stream_has_enough_packets(is->video_st, is->video_stream, &is->videoq))) {
            /* wait 10 ms */
            SDL_LockMutex(wait_mutex);
            SDL_CondWaitTimeout(is->continue_read_thread, wait_mutex, 10);
            SDL_UnlockMutex(wait_mutex);
            continue;
        }
        if (!is->paused &&
            (!is->video_st || (is->viddec.finished == is->videoq.serial && frame_queue_nb_remaining(&is->pictq) == 0))) {
            if (loop != 1 && (!loop || --loop)) {
                stream_seek(is, start_time != AV_NOPTS_VALUE ? start_time : 0, 0, 0);
            } else if (autoexit) {
                ret = AVERROR_EOF;
                goto fail;
            }
        }
        ret = av_read_frame(ic, pkt);
        if (ret < 0) {
            if ((ret == AVERROR_EOF || avio_feof(ic->pb)) && !is->eof) {
                if (is->video_stream >= 0)
                    packet_queue_put_nullpacket(&is->videoq, pkt, is->video_stream);
                is->eof = 1;
            }
            if (ic->pb && ic->pb->error) {
                if (autoexit)
                    goto fail;
                else
                    break;
            }
            SDL_LockMutex(wait_mutex);
            SDL_CondWaitTimeout(is->continue_read_thread, wait_mutex, 10);
            SDL_UnlockMutex(wait_mutex);
            continue;
        } else {
            is->eof = 0;
        }
        if (pkt->stream_index == is->video_stream &&
                   !(is->video_st->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
            packet_queue_put(&is->videoq, pkt);
        } else {
            av_packet_unref(pkt);
        }
    }

    ret = 0;
 fail:
    if (ic && !is->ic)
        avformat_close_input(&ic);

    av_packet_free(&pkt);
    if (ret != 0) {
        SDL_Event event;

        event.type = FF_QUIT_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);
    }
    SDL_DestroyMutex(wait_mutex);
    return 0;
}

static void print_render_info(VideoState *is) {
    int i;
    Uint32 window_pixel_format = SDL_GetWindowPixelFormat(is->window);
    av_log(NULL, AV_LOG_DEBUG, "SDL window pixel format: %s\n", SDL_GetPixelFormatName(window_pixel_format));

    av_log(NULL, AV_LOG_DEBUG, "SDL %s renderer supported format: ", is->renderer_info.name);
    for (i = 0; i < is->renderer_info.num_texture_formats; i++) {
        if (i > 0)
            av_log(NULL, AV_LOG_DEBUG, ", ");
        av_log(NULL, AV_LOG_DEBUG, "%s", SDL_GetPixelFormatName(is->renderer_info.texture_formats[i]));
    }
    av_log(NULL, AV_LOG_DEBUG, "\n");
}

static VideoState *stream_open(const char *filename, const AVInputFormat *iformat, int main_stream)
{
    VideoState *is;
    int ret;

    is = av_mallocz(sizeof(VideoState));
    if (!is)
        return NULL;
    is->filename = av_strdup(filename);
    if (!is->filename)
        goto fail;
    is->iformat = iformat;
    is->ytop    = 0;
    is->xleft   = 0;

    is->zoom_ratio = 1.0;
    is->window_attached_mode = 1;

    is->frame_queued = 0;
    is->frame_displayed = 0;

    /* start video display */
    if (frame_queue_init(&is->pictq, &is->videoq, VIDEO_PICTURE_QUEUE_SIZE, 1) < 0)
        goto fail;

    if (packet_queue_init(&is->videoq) < 0)
        goto fail;

    if (!(is->continue_read_thread = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
        goto fail;
    }

    init_clock(&is->vidclk, &is->videoq.serial);
    init_clock(&is->extclk, &is->extclk.serial);
    is->av_sync_type = av_sync_type;

    int flags = SDL_WINDOW_HIDDEN;
    if (alwaysontop)
#if SDL_VERSION_ATLEAST(2,0,5)
        flags |= SDL_WINDOW_ALWAYS_ON_TOP;
#else
        av_log(NULL, AV_LOG_WARNING, "Your SDL version doesn't support SDL_WINDOW_ALWAYS_ON_TOP. Feature will be inactive.\n");
#endif
    if (borderless)
        flags |= SDL_WINDOW_BORDERLESS;
    else
        flags |= SDL_WINDOW_RESIZABLE;

#ifdef SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR
    SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
#endif
    if (hwaccel && !enable_vulkan) {
        av_log(NULL, AV_LOG_INFO, "Enable vulkan renderer to support hwaccel %s\n", hwaccel);
        enable_vulkan = 1;
    }
    if (enable_vulkan) {
        is->vk_renderer = vk_get_renderer();
        if (is->vk_renderer) {
#if SDL_VERSION_ATLEAST(2, 0, 6)
            flags |= SDL_WINDOW_VULKAN;
#endif
        } else {
            av_log(NULL, AV_LOG_WARNING, "Doesn't support vulkan renderer, fallback to SDL renderer\n");
            enable_vulkan = 0;
        }
    }
    is->window = SDL_CreateWindow(program_name, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, default_width, default_height, flags);
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
    if (!is->window) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create window: %s", SDL_GetError());
        do_exit();
    }

    if (is->vk_renderer) {
        AVDictionary *dict = NULL;

        if (vulkan_params) {
            ret = av_dict_parse_string(&dict, vulkan_params, "=", ":", 0);
            if (ret < 0) {
                av_log(NULL, AV_LOG_FATAL, "Failed to parse, %s\n", vulkan_params);
                do_exit();
            }
        }
        ret = vk_renderer_create(is->vk_renderer, is->window, dict);
        av_dict_free(&dict);
        if (ret < 0) {
            av_log(NULL, AV_LOG_FATAL, "Failed to create vulkan renderer, %s\n", av_err2str(ret));
            do_exit();
        }
    } else {
        is->renderer = SDL_CreateRenderer(is->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!is->renderer) {
            av_log(NULL, AV_LOG_WARNING, "Failed to initialize a hardware accelerated renderer: %s\n", SDL_GetError());
            is->renderer = SDL_CreateRenderer(is->window, -1, 0);
        }
        if (is->renderer) {
            if (!SDL_GetRendererInfo(is->renderer, &is->renderer_info))
                av_log(NULL, AV_LOG_VERBOSE, "Initialized %s renderer.\n", is->renderer_info.name);
        }
        if (!is->renderer || !is->renderer_info.num_texture_formats) {
            av_log(NULL, AV_LOG_FATAL, "Failed to create window or renderer: %s", SDL_GetError());
            do_exit();
        }

        if (main_stream) {
            print_render_info(is);
        }
    }
    is->window_id = SDL_GetWindowID(is->window);

    is->read_tid     = SDL_CreateThread(read_thread, "read_thread", is);
    if (!is->read_tid) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateThread(): %s\n", SDL_GetError());
fail:
        stream_close(is);
        return NULL;
    }
    return is;
}

static void stream_cycle_channel(VideoState *is, int codec_type)
{
    AVFormatContext *ic = is->ic;
    int start_index, stream_index;
    int old_index;
    AVStream *st;
    AVProgram *p = NULL;
    int nb_streams = is->ic->nb_streams;

    if (codec_type == AVMEDIA_TYPE_VIDEO) {
        start_index = is->last_video_stream;
        old_index = is->video_stream;
    }
    stream_index = start_index;

    if (codec_type != AVMEDIA_TYPE_VIDEO && is->video_stream != -1) {
        p = av_find_program_from_stream(ic, NULL, is->video_stream);
        if (p) {
            nb_streams = p->nb_stream_indexes;
            for (start_index = 0; start_index < nb_streams; start_index++)
                if (p->stream_index[start_index] == stream_index)
                    break;
            if (start_index == nb_streams)
                start_index = -1;
            stream_index = start_index;
        }
    }

    for (;;) {
        if (++stream_index >= nb_streams)
        {
            if (start_index == -1)
                return;
            stream_index = 0;
        }
        if (stream_index == start_index)
            return;
        st = is->ic->streams[p ? p->stream_index[stream_index] : stream_index];
        if (st->codecpar->codec_type == codec_type) {
            /* check that parameters are OK */
            switch (codec_type) {
            case AVMEDIA_TYPE_VIDEO:
                goto the_end;
            default:
                break;
            }
        }
    }
 the_end:
    if (p && stream_index != -1)
        stream_index = p->stream_index[stream_index];
    av_log(NULL, AV_LOG_INFO, "Switch %s stream from #%d to #%d\n",
           av_get_media_type_string(codec_type),
           old_index,
           stream_index);

    stream_component_close(is, old_index);
    stream_component_open(is, stream_index);
}


static void toggle_full_screen(VideoState *is)
{
    is_full_screen = !is_full_screen;
    SDL_SetWindowFullscreen(is->window, is_full_screen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
    is->force_refresh = 1;
}

static void __move_by_motion(VideoState *is, double x, double y, int rel)
{
    is->window_attached_mode = 0;
    if (rel) {
        is->xleft += x;
        is->ytop += y;
    } else {
        is->xleft = x;
        is->ytop = y;
    }
    is->force_refresh = 1;
}

static void move_by_motion(VideoState *is, double x, double y, int rel)
{
    int i;
    if (is) {
        __move_by_motion(is, x, y, rel);
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        __move_by_motion(input_vs[i], x, y, rel);
    }
}

static void __zoom_stream(VideoState *is, double ratio, int rel)
{
    is->window_attached_mode = 0;

    if (rel) {
        is->zoom_ratio += ratio;
    } else {
        is->zoom_ratio = ratio;
    }
    is->zoom_ratio = FFMIN(FFMAX(is->zoom_ratio, 0.01), 8.0);
    is->force_refresh = 1;
}

static void zoom_stream(VideoState *is, double ratio, int rel)
{
    int i;
    if (is) {
        __zoom_stream(is, ratio, rel);
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        __zoom_stream(input_vs[i], ratio, rel);
    }
}

static void __restore_zoom_pos(VideoState *is)
{
    SDL_SetWindowSize(is->window, default_width, default_height);
    zoom_stream(is, 1.0, 0);
    move_by_motion(is, 0, 0, 0);
    is->window_attached_mode = 1;
}

static void restore_zoom_pos(VideoState *is)
{
    int i;
    if (is) {
        __restore_zoom_pos(is);
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        __restore_zoom_pos(input_vs[i]);
    }
}

static void __do_seek(VideoState *is, double incr)
{
    double pos;

    if (seek_by_bytes) {
        pos = -1;
        if (pos < 0 && is->video_stream >= 0)
            pos = frame_queue_last_pos(&is->pictq);
        if (pos < 0)
            pos = avio_tell(is->ic->pb);
        if (is->ic->bit_rate)
            incr *= is->ic->bit_rate / 8.0;
        else
            incr *= 180000.0;
        pos += incr;
    } else {
        pos = get_master_clock(is);
        if (isnan(pos))
            pos = (double)is->seek_pos / AV_TIME_BASE;
        pos += incr;
        if (is->ic->start_time != AV_NOPTS_VALUE && pos < is->ic->start_time / (double)AV_TIME_BASE)
            pos = is->ic->start_time / (double)AV_TIME_BASE;

        pos *= AV_TIME_BASE;
        incr *= AV_TIME_BASE;
    }

    stream_seek(is, (int64_t) pos, (int64_t) incr, seek_by_bytes);
}

static void do_seek(VideoState *is, double incr)
{
    int i;
    if (is) {
        __do_seek(is, incr);
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        __do_seek(input_vs[i], incr);
    }
}

static void do_sync(VideoState *is)
{
    double dpts = 0;
    Frame *vp;
    int i;
    vp = frame_queue_peek_last(&is->pictq);

    if (vp->frame->pts != AV_NOPTS_VALUE)
        dpts = av_q2d(is->video_st->time_base) * vp->frame->pts;
    else
        dpts = NAN;
    dpts *= AV_TIME_BASE;

    for (i = 0; i < nb_inputs; i++) {
        stream_seek(input_vs[i], (int64_t) dpts, 0, seek_by_bytes);
    }
}

static void __seek_by_motion(VideoState *is, double x)
{
    double pos, frac;

    if (seek_by_bytes || master_vs->ic->duration <= 0) {
        uint64_t size =  avio_size(master_vs->ic->pb);
        pos = size*x/master_vs->width;
    } else {
        int ns, hh, mm, ss;
        int tns, thh, tmm, tss;
        tns  = master_vs->ic->duration / 1000000LL;
        thh  = tns / 3600;
        tmm  = (tns % 3600) / 60;
        tss  = (tns % 60);
        frac = x / master_vs->width;
        ns   = frac * tns;
        hh   = ns / 3600;
        mm   = (ns % 3600) / 60;
        ss   = (ns % 60);
        av_log(NULL, AV_LOG_INFO,
               "Seek to %2.0f%% (%2d:%02d:%02d) of total duration (%2d:%02d:%02d)\n",
               frac*100, hh, mm, ss, thh, tmm, tss);
        pos = frac * master_vs->ic->duration;
        if (master_vs->ic->start_time != AV_NOPTS_VALUE)
            pos += master_vs->ic->start_time;
    }
    stream_seek(is, (int64_t) pos, 0, 0);
}

static int create_format_context(const char *filename, AVFormatContext **avf_ctx)
{
    const AVOutputFormat *fmt;
    AVStream *st;

    int ret;
    ret = avformat_alloc_output_context2(avf_ctx, NULL, NULL, filename);
    if (ret < 0 || NULL == *avf_ctx) {
        av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
        return ret;
    }
    fmt = (*avf_ctx)->oformat;
    if (fmt->video_codec == AV_CODEC_ID_NONE) {
        av_log(NULL, AV_LOG_ERROR, "Output format %s do not support video\n", fmt->name);
        return AVERROR_MUXER_NOT_FOUND;
    }

    st = avformat_new_stream((*avf_ctx), NULL);
    if (NULL == st) {
        av_log(NULL, AV_LOG_ERROR, "Could not allocate stream\n");
        return AVERROR(ENOMEM);
    }
    st->id = (*avf_ctx)->nb_streams - 1;

    return 0;
}

static int create_codec_context(const AVFrame *frame, const char *filename, const AVFormatContext *avf_ctx, AVCodecContext **avc_ctx)
{
    const AVOutputFormat *fmt = avf_ctx->oformat;
    enum AVCodecID codec_id;
    const AVCodec *codec;
    enum AVPixelFormat best;
    const enum AVPixelFormat *p;
    int ret;

    codec_id = av_guess_codec(fmt, NULL, filename, NULL, AVMEDIA_TYPE_VIDEO);
    if (AV_CODEC_ID_NONE == codec_id) {
        av_log(NULL, AV_LOG_ERROR, "Cannot guess codec from filename '%s'\n", filename);
        return AVERROR_ENCODER_NOT_FOUND;
    }
    codec = avcodec_find_encoder(codec_id);
    if (NULL == codec) {
        av_log(NULL, AV_LOG_ERROR, "Cannot get encoder '%s'\n", avcodec_get_name(codec_id));
        return AVERROR_ENCODER_NOT_FOUND;
    }
    av_log(NULL, AV_LOG_DEBUG, "Output format %s, encoder %s\n", fmt->name, codec->name);

    *avc_ctx = avcodec_alloc_context3(codec);
    if (NULL == *avc_ctx) {
        av_log(NULL, AV_LOG_ERROR, "Could not alloc an encoding context\n");
        return AVERROR(ENOMEM);
    }

    // choose best pixel format
    ret = avcodec_get_supported_config(*avc_ctx, NULL, AV_CODEC_CONFIG_PIX_FORMAT, 0, (const void **) &p, NULL);
    if (ret < 0)
        return ret;
    if (NULL == p) { // all possible pixel formats are supported
        best = frame->format;
    } else {
        best = avcodec_find_best_pix_fmt_of_list(p, frame->format, 0, NULL);
        if (AV_PIX_FMT_NONE == best) {
            av_log(NULL, AV_LOG_ERROR, "Could not find a suitable pixel format for '%s'\n",
                   avcodec_get_name(fmt->video_codec));
            avcodec_free_context(avc_ctx);
            return AVERROR_ENCODER_NOT_FOUND;
        }
    }
    (*avc_ctx)->pix_fmt = best;
    av_log(NULL, AV_LOG_DEBUG, "Choose pixel format %s for encoding\n", av_get_pix_fmt_name((*avc_ctx)->pix_fmt));
    (*avc_ctx)->width = frame->width;
    (*avc_ctx)->height = frame->height;
    (*avc_ctx)->time_base = (AVRational){1, 25};
    (*avc_ctx)->framerate = (AVRational){25, 1};

    /* Some formats want stream headers to be separate. */
    if (fmt->flags & AVFMT_GLOBALHEADER)
        (*avc_ctx)->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    (*avc_ctx)->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    ret = avcodec_open2(*avc_ctx, codec, NULL);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open codec!\n");
        avcodec_free_context(avc_ctx);
        return ret;
    }

    /* copy the stream parameters to the muxer */
    ret = avcodec_parameters_from_context(avf_ctx->streams[0]->codecpar, *avc_ctx);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Could not copy the stream parameters\n");
        avcodec_free_context(avc_ctx);
        return ret;
    }
    return 0;
}

static AVFrame *convert_frame(AVCodecContext *codec_ctx, const AVFrame *frame)
{
    AVFrame *oframe;
    struct SwsContext *sws_ctx;
    uint8_t *buffer;
    int buffer_size;

    oframe = av_frame_alloc();
    oframe->format = codec_ctx->pix_fmt;
    oframe->width = codec_ctx->width;
    oframe->height = codec_ctx->height;
    buffer_size = av_image_get_buffer_size(oframe->format, oframe->width,
                                           oframe->height, 1);
    buffer = av_malloc(buffer_size);
    av_image_fill_arrays(oframe->data, oframe->linesize, buffer,
                         oframe->format, oframe->width, oframe->height, 1);

    sws_ctx = sws_getContext(frame->width, frame->height, frame->format,
                             oframe->width, oframe->height, oframe->format,
                             SWS_SPLINE | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP | SWS_ACCURATE_RND,
                             0, 0, 0);
    if (NULL == sws_ctx) {
        av_log(NULL, AV_LOG_ERROR, "Cannot get sws_ctx.\n");
        av_frame_free(&oframe);
        return NULL;
    }
    sws_scale(sws_ctx, (const uint8_t * const*)frame->data, frame->linesize, 0,
              frame->height, oframe->data, oframe->linesize);

    sws_freeContext(sws_ctx);
    av_frame_copy_props(oframe, frame);
    return oframe;
}

static int encode_frame(AVFrame *frame, AVFormatContext *avf_ctx, AVCodecContext *avc_ctx)
{
    AVStream *st = avf_ctx->streams[0];
    AVFrame *oframe;
    AVPacket *pkt;
    int ret = 0;

    pkt = av_packet_alloc();
    if (NULL == pkt) {
        return AVERROR(ENOMEM);
    }

    if (avc_ctx->pix_fmt == frame->format) {
        oframe = frame;
    } else {
        oframe = convert_frame(avc_ctx, frame);
        if (NULL == oframe) {
            return AVERROR_BUG;
        }
    }

    /* send the frame to the encoder */
    ret = avcodec_send_frame(avc_ctx, oframe);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Error sending a frame for encoding\n");
        goto fail;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(avc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error encoding a frame: %s\n", av_err2str(ret));
            break;
        }

        /* rescale output packet timestamp values from codec to stream timebase */
        av_packet_rescale_ts(pkt, avc_ctx->time_base, st->time_base);
        pkt->stream_index = st->index;

        /* Write the compressed frame to the media file. */
        ret = av_interleaved_write_frame(avf_ctx, pkt);
        /* pkt is now blank (av_interleaved_write_frame() takes ownership of
         * its contents and resets pkt), so that no unreferencing is necessary.
         * This would be different if one used av_write_frame(). */
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error while writing output packet: %s\n", av_err2str(ret));
            break;
        }
    }

fail:
    av_packet_free(&pkt);
    if (oframe != frame) {
        av_frame_free(&oframe);
    }

    return ret == AVERROR_EOF ? 1 : 0;
}

static int dump_frame(AVFrame *frame, const char *filename, const char *format)
{
    AVFormatContext *avf_ctx;
    AVCodecContext *avc_ctx;
    AVDictionary *options = NULL;
    int ret;

    ret = create_format_context(filename, &avf_ctx);
    if (ret < 0)
        return ret;
    avf_ctx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    ret = create_codec_context(frame, filename, avf_ctx, &avc_ctx);
    if (ret < 0)
        return ret;

    if (AV_LOG_INFO > av_log_get_level()) {
        av_dump_format(avf_ctx, 0, filename, 1);
    }

    ret = avio_open(&(avf_ctx->pb), filename, AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Could not open '%s': %s\n", filename, av_err2str(ret));
        return ret;
    }

    if (strcmp(avf_ctx->oformat->name, "image2") == 0) {
        // remove the warning from image2 muxer about writing a single image
        av_dict_set(&options, "update", "1", 0);
    }
    ret = avformat_write_header(avf_ctx, &options);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Error occurred when opening output file: %s\n", av_err2str(ret));
        return ret;
    }
    ret = encode_frame(frame, avf_ctx, avc_ctx);
    if (ret < 0)
        return ret;
    av_write_trailer(avf_ctx);

    avcodec_free_context(&avc_ctx);
    avio_closep(&(avf_ctx->pb));
    avformat_free_context(avf_ctx);
    return 0;
}

static int __save_frame(VideoState *is)
{
    Frame *vp;
    char filename[1024];
    double fps;
    int n;
    const char *ext;
    int ret;
    vp = frame_queue_peek_last(&is->pictq);

    fps = av_q2d(is->video_st->avg_frame_rate);
    n = av_q2d(is->video_st->time_base) * vp->frame->pts * fps;

    ext = (save_format == NULL) ? "png" : save_format;
    snprintf(filename, sizeof(filename), "%s.%04d.%s", is->filename, n, ext);
    ret = dump_frame(vp->frame, filename, ext);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Encode frame failed!\n");
    } else {
        av_log(NULL, AV_LOG_INFO, "Write frame %3"PRId64" to %s\n", vp->frame->pts, filename);
    }
    return ret;
}

static void save_frame(VideoState *is)
{
    int i;
    int ret = 0;

    if (is) {
        ret = __save_frame(is);
        goto finish;
    }
    for (i = 0; i < nb_inputs; i++) {
        ret += __save_frame(input_vs[i]);
    }

finish:
    if (ret) {
        av_log(NULL, AV_LOG_ERROR, "Save frame failed!\n");
        exit(1);
    }
    return;
}

static void seek_by_motion(VideoState *is, double x)
{
    int i;
    if (is) {
        __seek_by_motion(is, x);
        return;
    }
    for (i = 0; i < nb_inputs; i++) {
        __seek_by_motion(input_vs[i], x);
    }
}

static int need_refresh(void)
{
    VideoState *is;
    int i;
    int playing = 0;
    int force_refresh = 0;

    for (i = 0; i < nb_inputs; i++) {
        is = input_vs[i];
        playing |= !is->paused;
        force_refresh |= is->force_refresh;
    }

    return playing || force_refresh;
}

static void refresh_loop_wait_event(SDL_Event *event) {
    double remaining_time = 0.0;
    SDL_PumpEvents();
    while (!SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT)) {
        if (remaining_time > 0.0)
            av_usleep((int64_t)(remaining_time * 1000000.0));
        remaining_time = REFRESH_RATE;
        if (need_refresh())
            video_refresh(&remaining_time);
        SDL_PumpEvents();
    }
}

static VideoState *get_state_from_window_id(int window_id)
{
    VideoState *is = NULL;
    int i;

    for (i = 0; i < nb_inputs; i++) {
        is = input_vs[i];
        if (is->window_id == window_id) {
            break;
        }
    }
    return is;
}

/* handle an event sent by the GUI */
static void event_loop(void)
{
    VideoState *cur_stream;
    SDL_Event event;
    SDL_Keymod keymod;
    double incr, ratio_rel;

    step_to_next_frame(NULL); // paused at first frame
    for (;;) {
        refresh_loop_wait_event(&event);
        cur_stream = get_state_from_window_id(event.key.windowID);
        keymod = SDL_GetModState();
        switch (event.type) {
        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
                do_exit();
                break;
            }
            // If we don't yet have a window, skip all key events, because read_thread might still be initializing...
            if (!master_vs->width)
                continue;
            switch (event.key.keysym.sym) {
            case SDLK_f:
                toggle_full_screen(cur_stream);
                break;
            case SDLK_SPACE:
                toggle_pause(cur_stream, (keymod & KMOD_SHIFT) ? 0 : 1);
                break;
            case SDLK_s: // S: Step to next frame
                if (keymod & KMOD_CTRL) {
                    save_frame((keymod & KMOD_SHIFT) ? cur_stream : NULL);
                } else {
                    step_to_next_frame((keymod & KMOD_SHIFT) ? cur_stream : NULL);
                }
                break;
            case SDLK_a:
                stream_cycle_channel(cur_stream, AVMEDIA_TYPE_VIDEO);
                break;
            case SDLK_v:
                stream_cycle_channel(cur_stream, AVMEDIA_TYPE_VIDEO);
                break;
            case SDLK_c:
                do_sync(cur_stream);
                break;
            case SDLK_w:
                if (cur_stream->vfilter_idx < nb_vfilters - 1) {
                    if (++cur_stream->vfilter_idx >= nb_vfilters)
                        cur_stream->vfilter_idx = 0;
                } else {
                    cur_stream->vfilter_idx = 0;
                }
                break;
            case SDLK_z:
                restore_zoom_pos((keymod & KMOD_SHIFT) ? cur_stream : NULL);
                break;
            case SDLK_MINUS:
                if (keymod & KMOD_CTRL) {
                    zoom_stream((keymod & KMOD_SHIFT) ? cur_stream : NULL, -0.2, 1);
                }
                break;
            case SDLK_EQUALS:
                if (keymod & KMOD_CTRL) {
                    zoom_stream((keymod & KMOD_SHIFT) ? cur_stream : NULL, 0.2, 1);
                }
                break;
            case SDLK_PAGEUP:
                incr = 600.0;
                goto do_seek;
            case SDLK_PAGEDOWN:
                incr = -600.0;
                goto do_seek;
            case SDLK_LEFT:
                incr = seek_interval ? -seek_interval : -1.0;
                goto do_seek;
            case SDLK_RIGHT:
                incr = seek_interval ? seek_interval : 1.0;
                goto do_seek;
            case SDLK_UP:
                incr = 60.0;
                goto do_seek;
            case SDLK_DOWN:
                incr = -60.0;
            do_seek:
                do_seek((keymod & KMOD_SHIFT) ? cur_stream : NULL, incr);
                break;
            default:
                break;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                static int64_t last_mouse_left_click = 0;
                if (av_gettime_relative() - last_mouse_left_click <= 500000) {
                    toggle_full_screen(cur_stream);
                    last_mouse_left_click = 0;
                } else {
                    last_mouse_left_click = av_gettime_relative();
                }
            }
        case SDL_MOUSEMOTION:
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    seek_by_motion((keymod & KMOD_SHIFT) ? cur_stream : NULL,
                            event.button.x);
                } else if (event.motion.state & SDL_BUTTON_MMASK) {
                    move_by_motion((keymod & KMOD_SHIFT) ? cur_stream : NULL,
                            0, 0, 1);
                }
            } else {
                if (event.motion.state & SDL_BUTTON_RMASK) {
                    seek_by_motion((keymod & KMOD_SHIFT) ? cur_stream : NULL,
                            event.motion.x);
                } else if (event.motion.state & SDL_BUTTON_MMASK) {
                    move_by_motion((keymod & KMOD_SHIFT) ? cur_stream : NULL,
                            event.motion.xrel, event.motion.yrel, 1);
                }
            }
            break;
        case SDL_MOUSEWHEEL:
            if (SDL_MOUSEWHEEL_NORMAL == event.wheel.direction) {
                ratio_rel = event.wheel.y / 20.0;
            } else {
                ratio_rel = -event.wheel.y / 20.0;
            }
            zoom_stream((keymod & KMOD_SHIFT) ? cur_stream : NULL, ratio_rel, 1);
            break;
        case SDL_WINDOWEVENT:
            switch (event.window.event) {
                case SDL_WINDOWEVENT_SIZE_CHANGED:
                    screen_width  = cur_stream->width  = event.window.data1;
                    screen_height = cur_stream->height = event.window.data2;
                    if (cur_stream->vis_texture) {
                        SDL_DestroyTexture(cur_stream->vis_texture);
                        cur_stream->vis_texture = NULL;
                    }
                case SDL_WINDOWEVENT_EXPOSED:
                    cur_stream->force_refresh = 1;
            }
            break;
        case SDL_QUIT:
        case FF_QUIT_EVENT:
            do_exit();
            break;
        default:
            break;
        }
    }
}

static int opt_width(void *optctx, const char *opt, const char *arg)
{
    double num;
    int ret = parse_number(opt, arg, OPT_TYPE_INT64, 1, INT_MAX, &num);
    if (ret < 0)
        return ret;

    screen_width = num;
    return 0;
}

static int opt_height(void *optctx, const char *opt, const char *arg)
{
    double num;
    int ret = parse_number(opt, arg, OPT_TYPE_INT64, 1, INT_MAX, &num);
    if (ret < 0)
        return ret;

    screen_height = num;
    return 0;
}

static int opt_format(void *optctx, const char *opt, const char *arg)
{
    file_iformat = av_find_input_format(arg);
    if (!file_iformat) {
        av_log(NULL, AV_LOG_FATAL, "Unknown input format: %s\n", arg);
        return AVERROR(EINVAL);
    }
    return 0;
}

static int opt_sync(void *optctx, const char *opt, const char *arg)
{
    if (!strcmp(arg, "video"))
        av_sync_type = AV_SYNC_VIDEO_MASTER;
    else if (!strcmp(arg, "ext"))
        av_sync_type = AV_SYNC_EXTERNAL_CLOCK;
    else {
        av_log(NULL, AV_LOG_ERROR, "Unknown value for %s: %s\n", opt, arg);
        exit(1);
    }
    return 0;
}

static int opt_input_file(void *optctx, const char *filename)
{
    if (nb_inputs >= MAX_INPUT_NUM) {
        av_log(NULL, AV_LOG_FATAL,
               "Current only support %d inputs in max.\n", MAX_INPUT_NUM);
        return AVERROR(EINVAL);
    }
    if (!strcmp(filename, "-"))
        filename = "fd:";
    input_filenames[nb_inputs] = av_strdup(filename);
    if (!input_filenames[nb_inputs])
        return AVERROR(ENOMEM);
    nb_inputs++;

    return 0;
}

static int dummy;

static const OptionDef options[] = {
    CMDUTILS_COMMON_OPTIONS
    { "x",                  OPT_TYPE_FUNC, OPT_FUNC_ARG, { .func_arg = opt_width }, "force displayed width", "width" },
    { "y",                  OPT_TYPE_FUNC, OPT_FUNC_ARG, { .func_arg = opt_height }, "force displayed height", "height" },
    { "fs",                 OPT_TYPE_BOOL,            0, { &is_full_screen }, "force full screen" },
    { "vst",                OPT_TYPE_STRING, OPT_EXPERT, { &wanted_stream_spec[AVMEDIA_TYPE_VIDEO] }, "select desired video stream", "stream_specifier" },
    { "ss",                 OPT_TYPE_TIME,            0, { &start_time }, "seek to a given position in seconds", "pos" },
    { "bytes",              OPT_TYPE_INT,             0, { &seek_by_bytes }, "seek by bytes 0=off 1=on -1=auto", "val" },
    { "seek_interval",      OPT_TYPE_FLOAT,           0, { &seek_interval }, "set seek interval for left/right keys, in seconds", "seconds" },
    { "noborder",           OPT_TYPE_BOOL,            0, { &borderless }, "borderless window" },
    { "alwaysontop",        OPT_TYPE_BOOL,            0, { &alwaysontop }, "window always on top" },
    { "f",                  OPT_TYPE_FUNC, OPT_FUNC_ARG, { .func_arg = opt_format }, "force format", "fmt" },
    { "stats",              OPT_TYPE_BOOL,   OPT_EXPERT, { &show_status }, "show status", "" },
    { "fast",               OPT_TYPE_BOOL,   OPT_EXPERT, { &fast }, "non spec compliant optimizations", "" },
    { "genpts",             OPT_TYPE_BOOL,   OPT_EXPERT, { &genpts }, "generate pts", "" },
    { "drp",                OPT_TYPE_INT,    OPT_EXPERT, { &decoder_reorder_pts }, "let decoder reorder pts 0=off 1=on -1=auto", "" },
    { "lowres",             OPT_TYPE_INT,    OPT_EXPERT, { &lowres }, "", "" },
    { "sync",               OPT_TYPE_FUNC, OPT_FUNC_ARG | OPT_EXPERT, { .func_arg = opt_sync }, "set videos sync. type (type=video/ext)", "type" },
    { "autoexit",           OPT_TYPE_BOOL,   OPT_EXPERT, { &autoexit }, "exit at the end", "" },
    { "loop",               OPT_TYPE_INT,    OPT_EXPERT, { &loop }, "set number of times the playback shall be looped", "loop count" },
    { "framedrop",          OPT_TYPE_BOOL,   OPT_EXPERT, { &framedrop }, "drop frames when cpu is too slow", "" },
    { "infbuf",             OPT_TYPE_BOOL,   OPT_EXPERT, { &infinite_buffer }, "don't limit the input buffer size (useful with realtime streams)", "" },
    { "left",               OPT_TYPE_INT,    OPT_EXPERT, { &screen_left }, "set the x position for the left of the window", "x pos" },
    { "top",                OPT_TYPE_INT,    OPT_EXPERT, { &screen_top }, "set the y position for the top of the window", "y pos" },
    { "vf",                 OPT_TYPE_FUNC, OPT_FUNC_ARG | OPT_EXPERT, { .func_arg = opt_add_vfilter }, "set video filters", "filter_graph" },
    { "i",                  OPT_TYPE_BOOL,            0, { &dummy}, "read specified file", "input_file"},
    { "vcodec",             OPT_TYPE_STRING, OPT_EXPERT, { &video_codec_name }, "force video decoder",    "decoder_name" },
    { "autorotate",         OPT_TYPE_BOOL,            0, { &autorotate }, "automatically rotate video", "" },
    { "find_stream_info",   OPT_TYPE_BOOL, OPT_INPUT | OPT_EXPERT, { &find_stream_info },
        "read and decode the streams to fill missing information with heuristics" },
    { "filter_threads",     OPT_TYPE_INT,    OPT_EXPERT, { &filter_nbthreads }, "number of filter threads per graph" },
    { "enable_vulkan",      OPT_TYPE_BOOL,            0, { &enable_vulkan }, "enable vulkan renderer" },
    { "vulkan_params",      OPT_TYPE_STRING, OPT_EXPERT, { &vulkan_params }, "vulkan configuration using a list of key=value pairs separated by ':'" },
    { "hwaccel",            OPT_TYPE_STRING, OPT_EXPERT, { &hwaccel }, "use HW accelerated decoding" },
    { "save_format",        OPT_TYPE_STRING,          0, { &save_format }, "format of saved frames, default is png" },
    { "use_10bit",          OPT_TYPE_INT,    OPT_EXPERT, { &use_10bit }, "whether to use 10 bit depth for rendering, 0=off 1=on -1=auto", "" },
    { "no_colorspace_hint", OPT_TYPE_BOOL,   OPT_EXPERT, { &no_colorspace_hint }, "passthrough color components as used \"as is\", only affect vulkan render", "" },
    { NULL, },
};

static void show_usage(void)
{
    av_log(NULL, AV_LOG_INFO, "Simple video comparing tool\n");
    av_log(NULL, AV_LOG_INFO, "usage: %s [options] input_files\n", program_name);
    av_log(NULL, AV_LOG_INFO, "\n");
}

void show_help_default(const char *opt, const char *arg)
{
    av_log_set_callback(log_callback_help);
    show_usage();
    show_help_options(options, "Main options:", 0, OPT_EXPERT);
    show_help_options(options, "Advanced options:", OPT_EXPERT, 0);
    printf("While playing:\n"
           "q, ESC              quit\n"
           "SPACE               pause/start playing\n"
           "s                   activate frame-step mode in all windows\n"
           "left/right          seek backward/forward 10 seconds in all windows or to custom interval if -seek_interval is set\n"
           "down/up             seek backward/forward 1 minute in all windows\n"
           "page down/up        seek backward/forward 10 minutes in all windows\n"
           "right mouse click   seek to percentage corresponding to fraction of width in all windows\n"
           "c                   sync all videos' playing time\n"
           "v                   cycle video channel in the current window\n"
           "w                   cycle video filters in the current window\n"
           "CTRL s              save frames in all windows to png format or to custom format if -save_format is set\n"
           "f                   toggle current window to full screen\n"
           "left double-click   toggle current window to full screen\n"
           "scroll mouse        zoom in/out picture in all windows\n"
           "CTRL -              zoom out picture in all windows\n"
           "CTRL =              zoom in picture in all windows\n"
           "middle mouse click  move picture position in all windows\n"
           "                    note: when you move or zoom a window, that window's picture size would be detached with window\n"
           "z                   restore window size, zoom ratio, picture position and detached state for all windows\n"
           "SHIFT               if SHIFT is also pressed, control would only apply to current window\n"
           );
}

/* Called from the main */
int main(int argc, char **argv)
{
    int flags, ret;
    int i;

    init_dynload();

    av_log_set_flags(AV_LOG_SKIP_REPEATED);
    parse_loglevel(argc, argv, options);

    /* register all codecs, demux and protocols */
#if CONFIG_AVDEVICE
    avdevice_register_all();
#endif
    avformat_network_init();

    signal(SIGINT , sigterm_handler); /* Interrupt (ANSI).    */
    signal(SIGTERM, sigterm_handler); /* Termination (ANSI).  */

    show_banner(argc, argv, options);

    ret = parse_options(NULL, argc, argv, options, opt_input_file);
    if (ret < 0)
        exit(ret == AVERROR_EXIT ? 0 : 1);
    if (!nb_inputs) {
        show_usage();
        av_log(NULL, AV_LOG_FATAL, "An input file must be specified\n");
        av_log(NULL, AV_LOG_FATAL, "Use -h to get full help\n");
        exit(1);
    }

    flags = SDL_INIT_VIDEO | SDL_INIT_TIMER;
    if (SDL_Init (flags)) {
        av_log(NULL, AV_LOG_FATAL, "Could not initialize SDL - %s\n", SDL_GetError());
        av_log(NULL, AV_LOG_FATAL, "(Did you set the DISPLAY variable?)\n");
        exit(1);
    }

    SDL_EventState(SDL_SYSWMEVENT, SDL_IGNORE);
    SDL_EventState(SDL_USEREVENT, SDL_IGNORE);

    for (i = 0; i < nb_inputs; i++) {
        input_vs[i] = stream_open(input_filenames[i], file_iformat, i == 0);
        input_vs[i]->index = i;
        if (!input_vs[i]) {
            av_log(NULL, AV_LOG_FATAL, "Failed to initialize VideoState!\n");
            do_exit();
        }
        av_usleep((int64_t)(0.100 * 1000000.0)); // sleep 100ms wait stream finish init
    }
    master_vs = input_vs[0];
    event_loop();

    /* never returns */

    return 0;
}

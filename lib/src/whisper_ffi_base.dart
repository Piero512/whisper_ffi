import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:whisper_ffi/src/whisper_bindings.dart';
import 'package:whisper_ffi/src/whisper_models.dart';

typedef WhisperProgressCb = Void Function(
  Pointer<whisper_context>,
  Pointer<whisper_state>,
  Int,
  Pointer<Void>,
);
typedef WhisperProgressNativeDartCb = void Function(
  Pointer<whisper_context>,
  Pointer<whisper_state>,
  int,
  Pointer<Void>,
);
typedef WhisperNewSegmentCb = Void Function(
  Pointer<whisper_context>,
  Pointer<whisper_state>,
  Int,
  Pointer<Void>,
);
typedef WhisperNewSegmentNativeDartCb = void Function(
  Pointer<whisper_context>,
  Pointer<whisper_state>,
  int,
  Pointer<Void>,
);

typedef GetNewStateCallback = Pointer<whisper_state> Function();

/// Wraps whisper bindings to allow a model
class WhisperModel implements Finalizable {
  static NativeFinalizer? _finalizer;
  final Pointer<whisper_context> ctx;
  final Pointer<whisper_full_params> params;
  final WhisperBindings wFfi;
  bool disposed = false;

  WhisperModel._(this.wFfi, this.ctx, this.params);

  factory WhisperModel.fromPath(WhisperBindings ffi, String pathToModel,
      {String language = 'es', String? prompt}) {
    return using((alloc) {
      _finalizer ??= NativeFinalizer(ffi.addresses.whisper_free.cast());
      final modelCtx = ffi.whisper_init_from_file(
          pathToModel.toNativeUtf8(allocator: alloc).cast());
      if (modelCtx == nullptr) {
        throw ArgumentError.value(pathToModel, 'pathToModel');
      }
      final params = ffi.whisper_full_default_params_by_ref(
        whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH,
      );
      params.ref.language = language.toNativeUtf8().cast();
      if (prompt != null) {
        params.ref.initial_prompt = prompt.toNativeUtf8().cast();
      }
      final whisperModel = WhisperModel._(ffi, modelCtx, params);
      _finalizer?.attach(
        whisperModel,
        modelCtx.cast(),
        detach: whisperModel,
        externalSize: File(pathToModel).statSync().size,
      );
      return whisperModel;
    });
  }

  String whisperPrintSysInfo() {
    return wFfi.whisper_print_system_info().cast<Utf8>().toDartString();
  }

  Iterable<String> retrieveAllSegments() sync* {
    for (var i = 0; i < segmentCount; i++) {
      yield wFfi
          .whisper_full_get_segment_text(ctx, i)
          .cast<Utf8>()
          .toDartString();
    }
  }

  void whisperFull(
    Pointer<Float> samples,
    int sampleCount, {
    NativeCallable<WhisperProgressCb>? progressCb,
    Pointer<Void>? progressCbData,
    NativeCallable<WhisperNewSegmentCb>? newSegmentCb,
    Pointer<Void>? newSegmentCbData,
  }) {
    if (progressCb != null) {
      final ptr = progressCb.nativeFunction;
      assert(ptr != nullptr);
      params.ref.progress_callback = ptr.cast();
    }
    if (progressCbData != null) {
      params.ref.progress_callback_user_data = progressCbData;
    }
    if (newSegmentCb != null) {
      final ptr = newSegmentCb.nativeFunction;
      assert(ptr != nullptr);
      params.ref.new_segment_callback = ptr.cast();
    }
    if (newSegmentCbData != null) {
      params.ref.new_segment_callback_user_data = newSegmentCbData;
    }
    final retval = wFfi.whisper_full(ctx, params.ref, samples, sampleCount);
    if (retval != 0) {
      print("Negative value received from whisper_full");
    }
  }

  int get segmentCount => wFfi.whisper_full_n_segments(ctx);

  List<WhisperTranscriptionPart> getAllSegments() {
    return retrieveTranscriptionParts(0, segmentCount).toList();
  }

  Iterable<WhisperTranscriptionPart> retrieveTranscriptionParts(
    int fromSegment,
    int toSegment,
  ) sync* {
    for (int i = fromSegment; i < toSegment; i++) {
      final start = wFfi.whisper_full_get_segment_t0(ctx, i);
      final end = wFfi.whisper_full_get_segment_t1(ctx, i);
      final transcriptPtr = wFfi.whisper_full_get_segment_text(ctx, i);
      final byteLength = transcriptPtr.cast<Utf8>().length;
      final decoder = Utf8Decoder(allowMalformed: true);
      final out =
          decoder.convert(transcriptPtr.cast<Uint8>().asTypedList(byteLength));

      yield WhisperTranscriptionPart(
        Duration(milliseconds: start * 10),
        Duration(milliseconds: end * 10),
        out,
      );
    }
  }

  Pointer<whisper_state> getNewState() {
    return wFfi.whisper_init_state(ctx);
  }

  void releaseStateList(List<Pointer<whisper_state>> allocatedPtrs) {
    for (final ptr in allocatedPtrs) {
      wFfi.whisper_free_state(ptr);
    }
  }

  T withNewStateCallback<T>(T Function(GetNewStateCallback) cb) {
    List<Pointer<whisper_state>> allocatedPtrs = [];
    allocator() {
      final ptr = getNewState();
      if (ptr != nullptr) {
        allocatedPtrs.add(ptr);
      }
      return ptr;
    }

    bool isAsync = false;
    try {
      final result = cb(allocator);
      if (result is Future) {
        isAsync = true;
        return (result.whenComplete(() => releaseStateList(allocatedPtrs))
            as T);
      }
      return result;
    } finally {
      if (!isAsync) {
        releaseStateList(allocatedPtrs);
      }
    }
  }

  void dispose() {
    if (!disposed) {
      disposed = true;
      _finalizer?.detach(this);
      malloc.free(params.ref.language);
      params.ref.language = nullptr;
      wFfi.whisper_free_params(params);
      wFfi.whisper_free(ctx);
    }
  }
}

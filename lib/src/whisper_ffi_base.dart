import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:whisper_ffi/src/whisper_bindings.dart';

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

/// Wraps whisper bindings to allow a model
class WhisperModel implements Finalizable {
  static NativeFinalizer? _finalizer;
  final Pointer<whisper_context> ctx;
  final Pointer<whisper_full_params> params;
  final WhisperBindings wFfi;
  bool disposed = false;

  WhisperModel._(this.wFfi, this.ctx, this.params);

  factory WhisperModel.fromPath(
    WhisperBindings ffi,
    String pathToModel, {
    String language = 'es',
  }) {
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
      params.ref.print_progress = true;
      // params.ref.speed_up = true;
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
    final segmentCount = wFfi.whisper_full_n_segments(ctx);
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

  List<String> getAllSegments() {
    final retval = <String>[];
    final genSegments = wFfi.whisper_full_n_segments(ctx);
    for (int i = 0; i < genSegments; i++) {
      final textPtr = wFfi.whisper_full_get_segment_text(ctx, i);
      retval.add(textPtr.cast<Utf8>().toDartString());
    }
    return retval;
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

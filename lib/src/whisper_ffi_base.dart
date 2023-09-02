import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:whisper_ffi/src/whisper_bindings.dart';

/// Wraps whisper bindings to allow a model
class WhisperModel implements Finalizable {
  static NativeFinalizer? _finalizer;
  final Pointer<whisper_context> ctx;
  final Pointer<whisper_full_params> params;
  final WhisperBindings ffi;
  bool disposed = false;

  WhisperModel._(this.ffi, this.ctx, this.params);

  static void progressCB(
    Pointer<whisper_context> ctx,
    Pointer<whisper_state> state,
    int progress,
    Pointer<Void> userData,
  ) {
    print("Progress: $progress");
  }

  static void newSegmentCB(
    Pointer<whisper_context> ctx,
    Pointer<whisper_state> state,
    int nSegment,
    Pointer<Void> userData,
  ) {
    print("New segment detected!");
    final getNewSegmentTextFn = userData
        .cast<
            NativeFunction<
                Pointer<Char> Function(Pointer<whisper_context>, Int)>>()
        .asFunction<Pointer<Char> Function(Pointer<whisper_context>, int)>();
    print(getNewSegmentTextFn(ctx, nSegment).cast<Utf8>().toDartString());
  }

  factory WhisperModel.fromPath(WhisperBindings ffi, String pathToModel,
      {String language = 'es'}) {
    return using((alloc) {
      _finalizer ??= NativeFinalizer(ffi.addresses.whisper_free.cast());
      final modelCtx = ffi.whisper_init_from_file(
          pathToModel.toNativeUtf8(allocator: alloc).cast());
      if (modelCtx == nullptr) {
        throw ArgumentError.value(pathToModel, 'pathToModel');
      }
      final params = calloc.call<whisper_full_params>();
      params.ref = ffi.whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH,
      );
      params.ref.progress_callback = Pointer.fromFunction(progressCB);
      params.ref.language = language.toNativeUtf8().cast();
      params.ref.new_segment_callback = Pointer.fromFunction(newSegmentCB);
      params.ref.new_segment_callback_user_data =
          ffi.addresses.whisper_full_get_segment_text.cast();
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
    return ffi.whisper_print_system_info().cast<Utf8>().toDartString();
  }

  Future<void> whisperFull(
    int samplesAddress,
    int sampleCount,
  ) async {
    final ctxAddr = ctx.address;
    final paramsAddr = params.address;
    return Isolate.run(() {
      _whisperFullSync(
          'whisper.dll', ctxAddr, paramsAddr, samplesAddress, sampleCount);
    }, debugName: 'whisper_full_inference');
  }

  static void _whisperFullSync(String pathToLibrary, int ctxIntPtr,
      int paramsIntPtr, int sampleIntPtr, int sampleCount) {
    using((a) {
      final samples = Pointer.fromAddress(sampleIntPtr).cast<Float>();
      final ctx = Pointer.fromAddress(ctxIntPtr).cast<whisper_context>();
      final params =
          Pointer.fromAddress(paramsIntPtr).cast<whisper_full_params>();
      final ffi = WhisperBindings(DynamicLibrary.open(pathToLibrary));
      final retval = ffi.whisper_full(ctx, params.ref, samples, sampleCount);
      if (retval != 0) {
        print("Negative value received from whisper_full");
      }
    });
  }

  List<String> getAllSegments() {
    return using((a) {
      final retval = <String>[];
      final genSegments = ffi.whisper_full_n_segments(ctx);
      for (int i = 0; i < genSegments; i++) {
        final textPtr = ffi.whisper_full_get_segment_text(ctx, i);
        retval.add(textPtr.cast<Utf8>().toDartString());
      }
      return retval;
    });
  }

  void dispose() {
    if (!disposed) {
      disposed = true;
      _finalizer?.detach(this);
      ffi.whisper_free(ctx);
    }
  }
}

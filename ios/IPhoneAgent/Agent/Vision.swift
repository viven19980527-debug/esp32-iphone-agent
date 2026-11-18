import CoreGraphics
import Foundation
import ImageIO
import Vision

/// Small image checks on screen frames: did the screen change, and where is the pointer. Port of vision.py.
enum Vision {
    /// Top share of the frame never compared: the status bar, where the Dynamic Island shows our Live Activity.
    static let topMask = 0.06

    /// A grayscale bitmap (PIL "L" weights), row-major.
    struct Gray {
        let w: Int, h: Int
        let px: [UInt8]
    }

    static func image(_ jpeg: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(jpeg as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// The image drawn at w x h (bilinear-ish scaling), converted to gray like PIL's convert("L").
    static func gray(_ img: CGImage, _ w: Int, _ h: Int) -> Gray {
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.interpolationQuality = .medium
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var px = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let r = Int(rgba[i * 4]), g = Int(rgba[i * 4 + 1]), b = Int(rgba[i * 4 + 2])
            px[i] = UInt8((r * 299 + g * 587 + b * 114 + 500) / 1000)
        }
        return Gray(w: w, h: h, px: px)
    }

    /// True when the screen content changed, including small things such as a checkmark.
    ///
    /// `ignore` lists screen points (fractions) where the pointer was or went: those spots are masked, so pointer
    /// moves never count. The status bar (clock) is ignored too.

    /// 把模型给的坐标吸附到最近的文字包围盒中心。
    /// 模型给的是"看图估的百分比坐标"，误差 1-3% 屏宽；iOS 按钮有效区又窄，
    /// 一次点空就要重新问模型（2-4 秒）。点空前先本地 OCR 校对一次，约 50-120 ms。
    static func snap(_ jpeg: Data, near p: CGPoint, search: CGFloat = 0.07) -> CGPoint? {
        guard let img = image(jpeg) else { return nil }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        let x = max(0, min(max(0, 1 - 2 * search), p.x - search))
        let y = max(0, min(max(0, 1 - 2 * search), 1 - p.y - search))   // Vision 原点在左下
        request.regionOfInterest = CGRect(x: x, y: y,
                                          width: min(1, 2 * search), height: min(1, 2 * search))

        let handler = VNImageRequestHandler(cgImage: img, options: [:])
        guard (try? handler.perform([request])) != nil else { return nil }
        guard let obs = request.results, !obs.isEmpty else { return nil }

        var best: (CGPoint, CGFloat)?
        for o in obs {
            let bb = o.boundingBox
            let c = CGPoint(x: bb.midX, y: 1 - bb.midY)
            let inside = abs(c.x - p.x) <= bb.width / 2 + 0.01 && abs(c.y - p.y) <= bb.height / 2 + 0.01
            if inside { return c }
            let d = hypot(c.x - p.x, c.y - p.y)
            if d <= search, best == nil || d < best!.1 { best = (c, d) }
        }
        return best?.0
    }

    static func changed(_ before: Data?, _ after: Data?, ignore: [CGPoint] = [], threshold: Double = 1.0) -> Bool {
        guard let before, let after, let a = image(before), let b = image(after) else { return true }
        if a.width != b.width || a.height != b.height { return true }
        let sa = gray(a, 32, 64), sb = gray(b, 32, 64)
        let total = zip(sa.px, sb.px).reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }
        if Double(total) / Double(sa.px.count) > threshold { return true }  // a big change: new screen, scrolled list
        let w = a.width / 2, h = a.height / 2  // fine pass for small changes, at half resolution
        let fa = gray(a, w, h), fb = gray(b, w, h)
        let masks = ignore.map { (Double($0.x) * Double(w), Double($0.y) * Double(h)) }
        let r2 = pow(0.05 * Double(w), 2)
        var hits = 0
        for i in 0..<(w * h) where abs(Int(fa.px[i]) - Int(fb.px[i])) > 25 {
            let x = Double(i % w), y = Double(i / w)
            if y > topMask * Double(h) && masks.allSatisfy({ pow(x - $0.0, 2) + pow(y - $0.1, 2) > r2 }) {
                hits += 1
                if hits >= 6 { return true }   // 够了就返回，别扫完整块图
            }
        }
        return false
    }

    /// Where the pointer is in `before`, given it moved right by `shift` (fraction of width) in `after`.
    ///
    /// Only the pointer moved between the two frames, so the changed pixels within `reach` of `near` are the
    /// pointer at both spots; their centroid sits shift/2 to the right of the first one. nil if unclear.
    static func pointer(_ before: Data, _ after: Data, near: CGPoint, shift: Double,
                        reach: CGPoint = CGPoint(x: 0.12, y: 0.08)) -> CGPoint? {
        guard let ia = image(before), let ib = image(after), ia.width == ib.width, ia.height == ib.height else {
            return nil
        }
        let w = ia.width, h = ia.height
        let a = gray(ia, w, h), b = gray(ib, w, h)
        let nx = Double(near.x), ny = Double(near.y), rx = Double(reach.x), ry = Double(reach.y)
        let left = max(0, Int((nx - rx) * Double(w)))
        let top = max(Int(0.05 * Double(h)), Int((ny - ry) * Double(h)))  // skip the status bar
        let right = min(w, Int((nx + rx) * Double(w))), bottom = min(h, Int((ny + ry) * Double(h)))
        guard right > left, bottom > top else { return nil }
        var hits: [(Int, Int)] = []
        for y in top..<bottom {
            for x in left..<right where abs(Int(a.px[y * w + x]) - Int(b.px[y * w + x])) > 30 {
                hits.append((x - left, y - top))
            }
        }
        let area = (right - left) * (bottom - top)
        guard hits.count >= 15, hits.count <= area / 4 else { return nil }  // nothing moved, or more than a pointer
        let xs = hits.map(\.0), ys = hits.map(\.1)
        if Double(xs.max()! - xs.min()!) > 0.15 * Double(w) || Double(ys.max()! - ys.min()!) > 0.07 * Double(h) {
            return nil  // not one compact spot
        }
        let cx = Double(xs.reduce(0, +)) / Double(hits.count) + Double(left)
        let cy = Double(ys.reduce(0, +)) / Double(hits.count) + Double(top)
        return CGPoint(x: cx / Double(w) - shift / 2, y: cy / Double(h))
    }
}

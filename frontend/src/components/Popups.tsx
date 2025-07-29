import { useContext } from "react";
import AppContext, { IR_LIMIT_PX } from "./interfaces";

import Modal from "react-bootstrap/Modal";
import Accordion from "react-bootstrap/Accordion";
import Toast from "react-bootstrap/Toast";
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from "react-bootstrap/Button";

export const ErrorMessage = () => {
  const {
    errorState: [errorObject, setErrorObject],
  } = useContext(AppContext)!;

  const handleClose = () => {
    setErrorObject({ msg: "", stackTrace: "" });
  };

  return (
    <>
      <Modal show={errorObject.msg !== ""} onHide={handleClose}>
        <Modal.Header
          style={{ backgroundColor: "#eb4034", color: "#ffffff" }}
          closeVariant="white"
          closeButton
        >
          <Modal.Title>Error</Modal.Title>
        </Modal.Header>
        <Modal.Body>{errorObject.msg}</Modal.Body>
        <Modal.Body>
          <Accordion defaultActiveKey="0">
            <Accordion.Item eventKey="1" key={1}>
              <Accordion.Header>Stack trace</Accordion.Header>
              {/*Need to manually overwrite the style here because of werid bug*/}
              <Accordion.Body style={{ visibility: "visible" }}>
                {errorObject.stackTrace}
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="dark" onClick={handleClose}>
            Understood!
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export const CLSModal = () => {
  const {
    analysisInfo: [analysisInfo],
    errVF: [errVF],
    showWarning: [showWarning, setShowWarning],
  } = useContext(AppContext)!;

  const hide = () => {
    setShowWarning("");
  };

  const getText = (state: "" | "cls" | "size" | "over") => {
    if (state == "cls") {
      return `Integral Range/feature size of ${analysisInfo?.integralRange.toFixed(2)} exceeds tested limit of ${IR_LIMIT_PX}px, results may be inaccurate.`;
    } else if (state == "size") {
      return `Image size < 200 px in at least one dimension. Results may be unstable.`;
    } else if (state == "over") {
      return `Image phase fraction uncertainty ${analysisInfo?.percentageErr.toFixed(2)}% already less than target uncertainty ${errVF.toFixed(2)}%`;
    } else {
      return "";
    }
  };

  const txt = showWarning == "over" ? "Good news!" : "Warning!";
  const bg = showWarning == "over" ? "#6ac40a" : "#fcba03";

  return (
    <>
      <ToastContainer className="p-5" position="bottom-start">
        <Toast onClose={(e) => hide()}>
          <Toast.Header
            className="roundedme-2"
            closeButton={true}
            style={{ backgroundColor: bg, color: "#ffffff" }}
          >
            <strong className="me-auto" style={{ fontSize: "1.5em" }}>
              {txt}
            </strong>
          </Toast.Header>
          <Toast.Body>{getText(showWarning)}</Toast.Body>
        </Toast>
      </ToastContainer>
    </>
  );
};

export const MoreInfo = () => {
  const {
    imageInfo: [imageInfo],
    analysisInfo: [analysisInfo],
    showInfo: [showInfo, setShowInfo],
    menuState: [menuState, setMenuState],
  } = useContext(AppContext)!;

  const handleClose = () => {
    setShowInfo(false);
    if (menuState == "conf_result_full") {
      setMenuState("conf_result");
    }
  };

  return (
    <>
      <Modal show={showInfo} onHide={handleClose} size="lg">
        <Modal.Header
          style={{ backgroundColor: "#212529", color: "#ffffff" }}
          closeVariant="white"
          closeButton
        >
          <Modal.Title>About the Model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            How sure are we that our measured phase fraction in a material
            represents the bulk phase fraction? We can never know for certain,
            but our model allows us to assess the uncertainty.
          </p>
          <p>
            We estimate the Two-Point-Correlation (TPC) - the probability a
            random vector starts and ends on pixels with the same value - of a
            chosen phase in a micrograph. From the TPC we determine the
            'Integral Range', which is effectively the feature size or
            'characteristic length scale' (CLS) of the phase in image.
            Previously this has been done statisically (and slowly), we do this
            directly by use of the FFT of the image.
          </p>
          <p>
            For finite-sized features in large images, the variance in the bulk
            phase fraction is some function of the CLS, so by determining it for
            our measured image we can establish uncertainty bounds on the bulk
            phase fraction given the measured phase fractions. Put another way,{" "}
            <b>
              we can state the material's phase fraction lies within some d% of
              the measured phase fraction with c% of the time.
            </b>
          </p>

          <p>
            Full details can be found in the{" "}
            <a href="https://arxiv.org/abs/2410.19568v1">paper</a>.
          </p>
          <p>
            Source code is available{" "}
            <a href="https://github.com/tldr-group/ImageRep">here</a>; please
            request any features or report any bugs in{" "}
            <a href="https://github.com/tldr-group/ImageRep/issues">
              the issues page!
            </a>
          </p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="dark" onClick={handleClose}>
            Understood!
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

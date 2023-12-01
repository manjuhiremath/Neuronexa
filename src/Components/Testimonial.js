import React from "react";
import ProfilePic from "../Assets/john-doe-image.png";
import { AiFillStar } from "react-icons/ai";

const Testimonial = () => {
  return (
    <div className="work-section-wrapper">
      <div className="work-section-top">
        <p className="primary-subheading">Testimonial</p>
        <h1 className="primary-heading">What They Are Saying</h1>
        <p className="primary-text">
        Neuronexa Labs has been instrumental in reshaping our online presence.
         Their dedicated team's commitment to excellence and proficiency in web development has significantly enhanced our
          brand's visibility and user experience. We're grateful
         for their innovative solutions and seamless collaboration, making them an indispensable partner for our digital journey.
        </p>
      </div>
      <div className="testimonial-section-bottom">
        <img src={ProfilePic} alt="" />
        <p>
        "Choosing Neuronexa was a game-changer for us. Their expertise, 
        attention to detail, and commitment to delivering a user-friendly and visually
         stunning website surpassed our expectations. Thanks to their innovative solutions, 
         our online presence has not only flourished but set new 
        industry standards. Neuronexa is our trusted ally in the digital landscape." - Wipro
        </p>
        <div className="testimonials-stars-container">
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
        </div>
        <h2>Manju Hiremath</h2>
      </div>
    </div>
  );
};

export default Testimonial;
